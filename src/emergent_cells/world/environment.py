"""A deterministic spatial environment with bounded resources and hazards.

The *simulator* centrally stores the grid for efficiency. Cells only observe
their own tile and four immediately adjacent tiles via local sensing.
No global state/labels are made available to a cell's policy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class WorldSpec:
    width: int = 64
    height: int = 64
    resource_patches: int = 14
    hazard_patches: int = 6
    patch_radius: float = 4.5
    regeneration: float = 0.022
    metabolism: float = 0.025
    movement_cost: float = 0.007
    hazard_damage: float = 0.09
    harvest_limit: float = 0.12
    starting_energy: float = 1.8
    maximum_energy: float = 4.0
    shock_step: int = -1

    def validate(self) -> None:
        if not 12 <= self.width <= 256 or not 12 <= self.height <= 256:
            raise ValueError("World dimensions must be 12..256")
        if not 0 <= self.resource_patches <= 128 or not 0 <= self.hazard_patches <= 128:
            raise ValueError("Patch counts must be 0..128")
        if not 0.5 <= self.patch_radius <= 32:
            raise ValueError("patch_radius must be 0.5..32")
        if not 0 <= self.regeneration <= 1:
            raise ValueError("regeneration must be 0..1")
        for name in ("metabolism", "movement_cost", "hazard_damage", "harvest_limit"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be 0..1")
        if not 0 < self.starting_energy <= self.maximum_energy <= 100:
            raise ValueError("0 < starting_energy <= maximum_energy <= 100 required")
        if self.shock_step != -1 and self.shock_step < 1:
            raise ValueError("shock_step must be -1 or >= 1")

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def _patch_field(rng: np.random.Generator, h: int, w: int,
                 count: int, radius: float) -> np.ndarray:
    """Local periodic Gaussian patches, capped as finite environmental capacity."""
    y, x = np.mgrid[0:h, 0:w]
    field = np.zeros((h, w), dtype=np.float64)
    for _ in range(count):
        px = rng.integers(w)
        py = rng.integers(h)
        dx = np.minimum(np.abs(x - px), w - np.abs(x - px))
        dy = np.minimum(np.abs(y - py), h - np.abs(y - py))
        field += np.exp(-(dx * dx + dy * dy) / (2 * radius**2))
    return np.clip(field, 0.0, 1.0)


class World:
    """Environment bookkeeping; cells do not access this state directly."""

    def __init__(self, spec: WorldSpec, seed: int):
        spec.validate()
        self.spec = spec
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.capacity = np.clip(
            0.012 + _patch_field(rng, spec.height, spec.width,
                                  spec.resource_patches, spec.patch_radius), 0, 1
        )
        self.resource = self.capacity.copy()
        self.hazard = _patch_field(
            rng, spec.height, spec.width, spec.hazard_patches,
            spec.patch_radius * 0.65
        )
        self.shock_applied = False

    def probe(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Read only cells' local tile coordinates, toroidal boundary."""
        ix = np.asarray(x, dtype=int) % self.spec.width
        iy = np.asarray(y, dtype=int) % self.spec.height
        return self.resource[iy, ix], self.hazard[iy, ix]

    def harvest(self, x: np.ndarray, y: np.ndarray,
                alive: np.ndarray) -> np.ndarray:
        """Allocate a finite tile resource fairly among all cells on that tile.

        Simultaneous harvest prevents iteration order giving a privileged cell
        more food; this is world physics, not a controller selecting actions.
        """
        count = x.size
        got = np.zeros(count, dtype=np.float64)
        idx = np.flatnonzero(alive)
        if not idx.size:
            return got
        linear = y[idx] * self.spec.width + x[idx]
        buckets = np.bincount(linear, minlength=self.spec.width * self.spec.height)
        stock = self.resource.ravel()
        portion = np.minimum(self.spec.harvest_limit,
                             stock[linear] / buckets[linear])
        got[idx] = portion
        used = np.bincount(linear, weights=portion, minlength=stock.size)
        stock -= used
        # Guard against floating-point roundoff, not an extra food source.
        np.maximum(stock, 0.0, out=stock)
        return got

    def step(self, step_index: int) -> None:
        if self.spec.shock_step == step_index and not self.shock_applied:
            # A reproducible resource-field displacement, not a reward label.
            shift = self.spec.width // 2
            self.capacity = np.roll(self.capacity, shift, axis=1)
            self.resource = np.roll(self.resource, shift, axis=1)
            self.shock_applied = True
        self.resource += self.spec.regeneration * (self.capacity - self.resource)
        np.clip(self.resource, 0.0, self.capacity, out=self.resource)

    def resource_total(self) -> float:
        return float(self.resource.sum())
