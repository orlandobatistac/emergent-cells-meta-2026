"""Local communication and plastic-link ablation on the existing cellular engine.

Scientific purpose: isolate whether cell-to-cell food cues improve learning and
whether locally modulated link strengths help beyond equal-count fixed links.
No global advantage/reward or per-cell policy optimizer is used.

This is an Oct 26 exploratory experiment and NOT eligible-period evidence
until organizers resolve the official Oct 27–Nov 4 building window.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np

from emergent_cells.cells import CellPopulation, CellSpec
from emergent_cells.world import World, WorldSpec


@dataclass(frozen=True)
class LinkSpec:
    max_links: int = 4
    radius: float = 5.0
    trust_learning_rate: float = 0.85
    minimum_trust: float = 0.55
    exclusion_steps: int = 8

    def validate(self) -> None:
        if not 1 <= self.max_links <= 32:
            raise ValueError("max_links must be 1..32")
        if not 0 < self.radius <= 32:
            raise ValueError("radius must be (0,32]")
        if not 0 < self.trust_learning_rate <= 10:
            raise ValueError("trust_learning_rate must be (0,10]")
        if not 0.1 <= self.minimum_trust < 1.0:
            raise ValueError("minimum_trust must be [0.1,1)")
        if not 1 <= self.exclusion_steps <= 1000:
            raise ValueError("exclusion_steps must be 1..1000")


class LinkedPopulation(CellPopulation):
    """Per-cell bounded, directed, local contacts.

    Cells may observe only alive peers within sensor radius. Each cell keeps
    its own limited slots and its own trust values. The link matrix is only
    a batch storage layout: row i is state belonging to cell i.
    """

    def __init__(self, world: WorldSpec, spec: CellSpec, seed: int,
                 link_spec: LinkSpec, plastic_links: bool):
        link_spec.validate()
        # Use exactly the same initial cell states as the baseline engine.
        super().__init__(world, spec, seed)
        n, k = spec.count, link_spec.max_links
        self.link_spec = link_spec
        self.plastic_links = plastic_links
        self.links = np.full((n, k), -1, dtype=np.int64)
        self.trust = np.ones((n, k), dtype=np.float64)
        self.slot_direction = np.zeros((n, k, 2), dtype=np.float64)
        self.slot_message = np.zeros((n, k), dtype=np.float64)
        self.last_excluded_target = np.full(n, -1, dtype=np.int64)
        self.exclude_until = np.zeros(n, dtype=np.int64)
        self.tick = 0
        self.contacts_created = 0
        self.contacts_expired = 0
        self.contacts_pruned = 0
        self.trust_update_count = 0

    def _social_features(self, world: World) -> tuple[np.ndarray, int]:
        n = self.spec.count
        w, h = world.spec.width, world.spec.height
        dx = self.x[None, :] - self.x[:, None]
        dy = self.y[None, :] - self.y[:, None]
        dx = (dx + w // 2) % w - w // 2
        dy = (dy + h // 2) % h - h // 2
        sq = dx * dx + dy * dy
        eligible = ((sq > 0) & (sq <= self.link_spec.radius**2)
                    & self.alive[:, None] & self.alive[None, :])
        self.slot_direction.fill(0)
        self.slot_message.fill(0)
        influence = np.zeros((n, 4), dtype=np.float64)
        active_contacts = 0

        for i in np.flatnonzero(self.alive):
            # Release contacts only when a neighbor leaves local range or dies.
            for slot, neighbor in enumerate(self.links[i]):
                if neighbor >= 0 and not eligible[i, neighbor]:
                    self.links[i, slot] = -1
                    self.trust[i, slot] = 1.0
                    self.contacts_expired += 1

            current = set(int(v) for v in self.links[i] if v >= 0)
            candidates = np.flatnonzero(eligible[i])
            # Tie-break on cell ID, not on hidden reward or any global score.
            order = candidates[np.lexsort((candidates, sq[i, candidates]))]
            for j in order:
                j = int(j)
                if j in current:
                    continue
                if (self.plastic_links and j == self.last_excluded_target[i]
                        and self.tick < self.exclude_until[i]):
                    continue
                free = np.flatnonzero(self.links[i] == -1)
                if not free.size:
                    break
                slot = int(free[0])
                self.links[i, slot] = j
                self.trust[i, slot] = 1.0
                self.contacts_created += 1
                current.add(j)

            active_slots = np.flatnonzero(self.links[i] >= 0)
            active_contacts += int(active_slots.size)
            ix, iy = 0.0, 0.0
            for slot in active_slots:
                neighbor = int(self.links[i, slot])
                distance = float(np.sqrt(sq[i, neighbor]))
                direction = np.array((dx[i, neighbor], dy[i, neighbor])) / distance
                message = float(self.last_harvest[neighbor])
                self.slot_direction[i, slot] = direction
                self.slot_message[i, slot] = message
                weight = self.trust[i, slot] * message / distance
                ix += float(direction[0]) * weight
                iy += float(direction[1]) * weight
            if active_slots.size:
                ix *= self.spec.social_signal_scale / active_slots.size
                iy *= self.spec.social_signal_scale / active_slots.size
            influence[i] = np.array((-iy, ix, iy, -ix))
        return np.clip(influence, -1.0, 1.0), active_contacts

    def step(self, world: World, rng: np.random.Generator,
             mode: str = "learned") -> dict:
        # Snapshot only each cell's own state before its environmental action.
        old_x = self.x.copy()
        old_y = self.y.copy()
        old_energy = self.energy.copy()
        old_baseline = self.baseline.copy()
        old_alive = self.alive.copy()
        state = super().step(world, rng, mode)
        if self.plastic_links and mode == "learned":
            w, h = world.spec.width, world.spec.height
            motion_x = (self.x - old_x + w // 2) % w - w // 2
            motion_y = (self.y - old_y + h // 2) % h - h // 2
            advantage = (self.energy - old_energy) - old_baseline
            alignment = (self.slot_direction[:, :, 0] * motion_x[:, None]
                         + self.slot_direction[:, :, 1] * motion_y[:, None])
            # A link gets local credit only for a *received food message*
            # pointing along or against the cell's own last movement.
            strength = np.minimum(self.slot_message / world.spec.harvest_limit,
                                  1.0) if world.spec.harvest_limit > 0 else np.zeros_like(self.slot_message)
            valid = (self.links >= 0) & old_alive[:, None] & (strength > 0)
            delta = (self.link_spec.trust_learning_rate
                     * advantage[:, None] * alignment * strength)
            self.trust[valid] += delta[valid]
            self.trust_update_count += int(np.count_nonzero(valid))
            np.clip(self.trust, 0.0, 2.5, out=self.trust)
            prunable = np.argwhere(valid & (self.trust < self.link_spec.minimum_trust))
            for i, slot in prunable:
                self.last_excluded_target[i] = self.links[i, slot]
                self.exclude_until[i] = self.tick + self.link_spec.exclusion_steps
                self.links[i, slot] = -1
                self.trust[i, slot] = 1.0
                self.contacts_pruned += 1
        self.tick += 1
        state.update({
            "active_links": int(np.count_nonzero(self.links >= 0)),
            "links_created": self.contacts_created,
            "links_expired": self.contacts_expired,
            "links_pruned": self.contacts_pruned,
            "trust_updates": self.trust_update_count
        })
        return state


def trial(seed: int, steps: int, condition: str,
          link_spec: LinkSpec = LinkSpec(), shock_step: int = 120,
          width: int = 64) -> dict:
    """One controlled episode: identical world, initial cells, random stream."""
    if condition not in {"none", "broadcast", "fixed", "adaptive"}:
        raise ValueError("Unknown connectivity condition")
    if not 1 <= steps <= 20000:
        raise ValueError("steps must be in 1..20000")
    ws = WorldSpec(width=width, height=width,
                   shock_step=shock_step if shock_step <= steps else -1)
    cs = CellSpec(count=256, communication_radius=0 if condition == "none" else
                  link_spec.radius)
    world = World(ws, seed)
    if condition in ("fixed", "adaptive"):
        cells = LinkedPopulation(ws, cs, seed=seed + 1000003,
                                 link_spec=link_spec,
                                 plastic_links=condition == "adaptive")
    else:
        cells = CellPopulation(ws, cs, seed=seed + 1000003)
    rng = np.random.default_rng(seed + 2000003)
    alive_cell_steps = 0
    history = []
    for tick in range(steps):
        state = cells.step(world, rng, "learned")
        alive_cell_steps += state["alive"]
        world.step(tick + 1)
        if (tick + 1) % max(1, steps // 12) == 0 or tick == 0:
            history.append({"step": tick + 1,
                            "survival_fraction": state["alive"] / cs.count,
                            "remaining_energy": state["total_energy"]})
        if not state["alive"]:
            break
    return {
        "seed": seed, "condition": condition, "steps_completed": tick + 1,
        "survival_fraction": float(cells.alive.mean()),
        "remaining_energy": float(cells.energy.sum()),
        "harvested_total": cells.total_harvested,
        "neighbor_messages": cells.total_messages,
        "alive_cell_steps": alive_cell_steps,
        "links_created": getattr(cells, "contacts_created", 0),
        "links_pruned": getattr(cells, "contacts_pruned", 0),
        "links_expired": getattr(cells, "contacts_expired", 0),
        "trust_updates": getattr(cells, "trust_update_count", 0),
        "shock_applied": world.shock_applied,
        "history": history,
    }


def benchmark(seeds: list[int], steps: int = 240,
              link_spec: LinkSpec = LinkSpec(), shock_step: int = 120,
              width: int = 64) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Unique seeds required")
    link_spec.validate()
    rows = [trial(seed, steps, condition, link_spec, shock_step, width)
            for seed in seeds
            for condition in ("none", "broadcast", "fixed", "adaptive")]
    summary = {}
    for condition in ("none", "broadcast", "fixed", "adaptive"):
        r = [row for row in rows if row["condition"] == condition]
        summary[condition] = {
            name: {
                "mean": float(np.mean([row[name] for row in r])),
                "std": float(np.std([row[name] for row in r]))
            }
            for name in ("survival_fraction", "remaining_energy",
                         "harvested_total", "neighbor_messages", "alive_cell_steps",
                         "links_created", "links_pruned", "trust_updates")
        }
    return {
        "scientific_status": "exploratory, not qualifying contest-period evidence",
        "hypothesis": "locally modulated neighbor connections can change adaptation",
        "limitations": [
            "256 cells are simulated using centrally scheduled NumPy operations.",
            "Weights receive cell-local reward-modulated updates, not physical energy.",
            "Fixed/adaptive have equal maximum link slots; broadcast is uncapped.",
            "This is an in-distribution diagnostic, not held-out generalization.",
            "No independent neural general intelligence is demonstrated.",
            "Created on Oct 26, while official guide lists Oct 27 start of building period."
        ],
        "parameters": {
            "seeds": seeds, "steps": steps,
            "shock_step": shock_step,
            "world_width": width, "link_spec": asdict(link_spec)
        },
        "summary": summary,
        "rows": rows
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--shock-step", type=int, default=120)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--max-links", type=int, default=4)
    p.add_argument("--radius", type=float, default=5.0)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    seeds = [int(i.strip()) for i in args.seeds.split(",") if i.strip()]
    result = benchmark(seeds, args.steps,
                       LinkSpec(max_links=args.max_links, radius=args.radius),
                       args.shock_step, args.width)
    text = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Saved {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
