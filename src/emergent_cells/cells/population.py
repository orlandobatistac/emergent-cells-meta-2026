"""Autonomous local cells: private energy, local sensing, neighbor messaging.

Cell-specific policy weights and homeostatic baselines are updated only from
that cell's own sensor/action and energy change. NumPy batching is an execution
optimization, not a central policy, centralized learning or global fitness.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from emergent_cells.world import World, WorldSpec


# Action vectors are physical grid moves, not labels of successful actions.
MOVES = np.array(((0, -1), (1, 0), (0, 1), (-1, 0)), dtype=np.int64)


@dataclass(frozen=True)
class CellSpec:
    count: int = 256
    communication_radius: float = 3.0
    learning_rate: float = 1.0
    reward_scale: float = 26.0
    temperature: float = 0.65
    social_signal_scale: float = 10.0
    sensor_scale: float = 6.0
    max_weight: float = 5.0

    def validate(self) -> None:
        if not 1 <= self.count <= 16384:
            raise ValueError("count must be 1..16384")
        if not 0 <= self.communication_radius <= 32:
            raise ValueError("communication_radius must be 0..32")
        if not 0 <= self.learning_rate <= 10 or not 0 <= self.reward_scale <= 100:
            raise ValueError("invalid learning rate/reward scale")
        if not 0.05 <= self.temperature <= 10:
            raise ValueError("temperature must be 0.05..10")


class CellPopulation:
    """Each row is the private state of one individual cell.

    The only inter-cell message consists of a neighboring cell's *previous*
    local food uptake. Spatial adjacency uses actual current positions.
    """

    def __init__(self, world: WorldSpec, spec: CellSpec, seed: int):
        spec.validate()
        self.spec = spec
        rng = np.random.default_rng(seed)
        self.x = rng.integers(0, world.width, size=spec.count, dtype=np.int64)
        self.y = rng.integers(0, world.height, size=spec.count, dtype=np.int64)
        self.energy = np.full(spec.count, world.starting_energy, dtype=np.float64)
        self.alive = np.ones(spec.count, dtype=bool)
        self.weights = rng.normal(0, 0.04, size=(spec.count, 3))
        self.baseline = np.zeros(spec.count, dtype=np.float64)
        self.last_harvest = np.zeros(spec.count, dtype=np.float64)
        self.total_harvested = 0.0
        self.total_hazard_cost = 0.0
        self.total_messages = 0

    def _social_features(self, world: World) -> tuple[np.ndarray, int]:
        n = self.spec.count
        if self.spec.communication_radius == 0:
            return np.zeros((n, 4), dtype=np.float64), 0

        # Coordinates represent actual cell positions; no topology-wide bus.
        # Each cell sees only neighbors no more than radius tiles away.
        w, h = world.spec.width, world.spec.height
        dx = self.x[None, :] - self.x[:, None]
        dy = self.y[None, :] - self.y[:, None]
        dx = (dx + w // 2) % w - w // 2
        dy = (dy + h // 2) % h - h // 2
        dist_sq = (dx * dx + dy * dy)
        mask = ((dist_sq <= self.spec.communication_radius ** 2)
                & (dist_sq > 0)
                & self.alive[:, None]
                & self.alive[None, :])
        strengths = (self.last_harvest[None, :]
                     * mask
                     / np.sqrt(np.maximum(dist_sq, 1)))
        nx = strengths * dx
        ny = strengths * dy
        peers = np.maximum(mask.sum(axis=1), 1)
        influence_x = nx.sum(axis=1) / peers * self.spec.social_signal_scale
        influence_y = ny.sum(axis=1) / peers * self.spec.social_signal_scale
        influence = MOVES[:, 0][None, :] * influence_x[:, None] + (
            MOVES[:, 1][None, :] * influence_y[:, None]
        )
        return np.clip(influence, -1.0, 1.0), int(mask.sum())

    def perceive(self, world: World) -> tuple[np.ndarray, int]:
        """Observations are local directional probes plus neighbor messages."""
        n = self.spec.count
        food0, hazard0 = world.probe(self.x, self.y)
        xnext = (self.x[:, None] + MOVES[:, 0]) % world.spec.width
        ynext = (self.y[:, None] + MOVES[:, 1]) % world.spec.height
        food1, hazard1 = world.probe(xnext, ynext)
        food_gradient = (food1 - food0[:, None]) * self.spec.sensor_scale
        danger_gradient = (hazard0[:, None] - hazard1) * self.spec.sensor_scale
        social, edges = self._social_features(world)
        features = np.stack((np.clip(food_gradient, -1, 1),
                             np.clip(danger_gradient, -1, 1),
                             social), axis=-1)
        assert features.shape == (n, 4, 3)
        return features, edges

    def step(self, world: World, rng: np.random.Generator,
             mode: str = "learned") -> dict:
        """One simultaneous local action/perception cycle.

        Modes:
          learned: per-cell local policy-gradient/homeostatic update
          frozen:  same initial weights and perceptions, no update
          random:  uniformly random action, no learning
        """
        if mode not in ("learned", "frozen", "random"):
            raise ValueError("Invalid mode")
        features, messages = self.perceive(world)
        self.total_messages += messages

        logits = np.einsum("naf,nf->na", features, self.weights)
        if mode == "random":
            logits[:] = 0.0
        logits /= self.spec.temperature
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        chosen = (np.cumsum(probs, axis=1) < rng.random((self.spec.count, 1))).sum(axis=1)
        chosen = np.minimum(chosen, 3)
        moving = self.alive.copy()
        self.x[moving] = (self.x[moving] + MOVES[chosen[moving], 0]) % world.spec.width
        self.y[moving] = (self.y[moving] + MOVES[chosen[moving], 1]) % world.spec.height

        harvested = world.harvest(self.x, self.y, moving)
        _, hazard_at_position = world.probe(self.x, self.y)
        hazard_cost = hazard_at_position * world.spec.hazard_damage * moving
        outgoing = (world.spec.metabolism + world.spec.movement_cost) * moving
        gain = harvested - outgoing - hazard_cost
        self.energy[moving] = np.clip(
            self.energy[moving] + gain[moving], 0.0, world.spec.maximum_energy
        )
        newly_dead = moving & (self.energy <= 0)
        self.alive[newly_dead] = False

        # REINFORCE gradient evaluated independently for every cell, using
        # only its own homeostatic energy change and own sensory features.
        if mode == "learned":
            expected = np.einsum("na,naf->nf", probs, features)
            selected = features[np.arange(self.spec.count), chosen]
            advantage = gain - self.baseline
            delta = (self.spec.learning_rate * self.spec.reward_scale
                     * advantage[:, None] * (selected - expected))
            self.weights[moving] += delta[moving]
            np.clip(self.weights, -self.spec.max_weight, self.spec.max_weight,
                    out=self.weights)
            self.baseline[moving] = (
                0.95 * self.baseline[moving] + 0.05 * gain[moving]
            )

        self.last_harvest[:] = harvested * self.alive
        self.total_harvested += float(harvested.sum())
        self.total_hazard_cost += float(hazard_cost.sum())
        return {
            "alive": int(self.alive.sum()),
            "total_energy": float(self.energy.sum()),
            "harvested": float(harvested.sum()),
            "hazard_cost": float(hazard_cost.sum()),
            "messages": messages,
            "mean_abs_weights": float(np.abs(self.weights).mean()),
        }
