"""Meta Model API -> validated WorldSpec -> independent cellular evaluation.

No model sees individual cell states, chooses individual actions, or updates
cell policies. The caller must explicitly select meta: NO silent mock fallback.
Oct 26 prototype predates guide's Oct 27- Nov 4 competition build period.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from typing import Any

from emergent_cells.cells import CellSpec
from emergent_cells.simulation import RunSpec, run
from emergent_cells.world import WorldSpec

META_MODEL = "muse-spark-1.3"
META_BASE_URL = "https://api.meta.ai/v1"

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "rationale": {"type": "string"},
        "resource_patches": {"type": "integer"},
        "hazard_patches": {"type": "integer"},
        "patch_radius": {"type": "number"},
        "regeneration": {"type": "number"},
        "hazard_damage": {"type": "number"},
        "shock_step": {"type": "integer"},
    },
    "required": ["name", "rationale", "resource_patches", "hazard_patches",
                 "patch_radius", "regeneration", "hazard_damage", "shock_step"],
    "additionalProperties": False,
}
PARAMS = ("resource_patches", "hazard_patches", "patch_radius",
          "regeneration", "hazard_damage", "shock_step")


class DesignError(ValueError):
    """Invalid or unavailable design. Never auto-fallback to a fake model."""


def validate_proposal(obj: Any) -> tuple[dict, WorldSpec]:
    if not isinstance(obj, dict) or set(obj) != set(PROPOSAL_SCHEMA["required"]):
        raise DesignError("Proposal must contain exactly the approved fields")
    if not isinstance(obj["name"], str) or not 1 <= len(obj["name"].strip()) <= 80:
        raise DesignError("name must be 1..80 characters")
    if not isinstance(obj["rationale"], str) or not 1 <= len(obj["rationale"].strip()) <= 300:
        raise DesignError("rationale must be 1..300 characters")
    ints = ("resource_patches", "hazard_patches", "shock_step")
    for key in PARAMS:
        value = obj[key]
        if key in ints:
            if type(value) is not int:
                raise DesignError(f"{key} must be an integer")
        elif type(value) not in (int, float) or not math.isfinite(value):
            raise DesignError(f"{key} must be a finite number")
    params = {key: obj[key] for key in PARAMS}
    try:
        world = WorldSpec(**params)
        world.validate()
    except (TypeError, ValueError) as ex:
        raise DesignError(f"World outside safe limits: {ex}") from ex
    if world.shock_step > 80:
        raise DesignError("shock_step must be -1 or <= 80 so it can be evaluated")
    normalized = {"name": obj["name"].strip(), "rationale": obj["rationale"].strip(),
                  **{k: int(params[k]) if k in ints else float(params[k]) for k in PARAMS}}
    return normalized, world


def _offline_template(prompt: str) -> dict:
    """Deterministic template selected by keywords: NOT a Meta request."""
    p = prompt.casefold()
    hard = any(x in p for x in ("escas", "dific", "sever", "hard", "scarce", "extremo"))
    danger = any(x in p for x in ("pelig", "amenaz", "hazard", "danger"))
    shock = any(x in p for x in ("cambi", "adapt", "shock", "perturb", "resilien"))
    cooperate = any(x in p for x in ("coop", "colabor", "comunic", "collectiv", "equipo"))
    proposed = {
        "name": "Mundo de prueba local (sin Meta)",
        "rationale": ("Plantilla de estrés de recursos y señales locales. "
                      "La cooperación no está garantizada por este motor de forrajeo."),
        "resource_patches": 5 if hard else 14,
        "hazard_patches": 12 if danger else 6,
        "patch_radius": 3.0 if hard else 4.5,
        "regeneration": 0.008 if hard else 0.022,
        "hazard_damage": 0.16 if danger else 0.09,
        "shock_step": 35 if shock else -1,
    }
    if cooperate:
        proposed["rationale"] = (
            "Plantilla de forrajeo con comunicación local, pero sin tarea que "
            "requiera matemáticamente cooperación. No es el experimento XOR.")
    return proposed


def _meta_proposal(prompt: str, client=None) -> tuple[dict, str | None]:
    if client is None:
        key = os.getenv("MODEL_API_KEY")
        if not key:
            raise DesignError(
                "MODEL_API_KEY no configurada. Usa 'plantilla local' "
                "o configura tu clave privada para seleccionar Meta.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise DesignError("Install optional dependency: pip install -e '.[meta]'") from exc
        client = OpenAI(api_key=key, base_url=META_BASE_URL,
                        timeout=20.0, max_retries=0)
    instruction = (
        "You design safe bounded experimental worlds for an existing "
        "decentralized digital cell simulator. Produce only JSON matching the "
        "given schema. Never direct individual cell actions, weights, rewards "
        "or produce executable code. Constraints: resource_patches 0..128; "
        "hazard_patches 0..128; patch_radius 0.5..32; regeneration 0..1; "
        "hazard_damage 0..1; shock_step -1 or 1..80. "
        "The cells have only local food/hazard sensors and simple neighbor "
        "messages. This foraging world CANNOT yet guarantee a task requiring "
        "cooperation. Disclose that limitation in rationale if requested. "
        "Do not predict or invent experimental results."
    )
    try:
        response = client.chat.completions.create(
            model=META_MODEL,
            messages=[{"role": "system", "content": instruction},
                      {"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "emergent_world_spec_v1",
                    "schema": PROPOSAL_SCHEMA,
                    "strict": True,
                },
            },
        )
        body = response.choices[0].message.content
        if not body:
            raise DesignError("Meta returned an empty proposal")
        return json.loads(body), getattr(response, "id", None)
    except DesignError:
        raise
    except (ValueError, TypeError) as exc:
        raise DesignError("Meta output could not be parsed or validated") from exc
    except Exception as exc:
        # Avoid exposing auth headers, key or other vendor diagnostic details.
        raise DesignError(f"Meta API request failed ({type(exc).__name__})") from exc


def design_world(prompt: str, source: str = "template", client=None) -> dict:
    if not isinstance(prompt, str) or not 4 <= len(prompt.strip()) <= 400:
        raise DesignError("Describe el desafío en 4..400 caracteres")
    if source not in ("template", "meta"):
        raise DesignError("Source must be explicitly 'template' or 'meta'")
    if source == "template":
        proposed, response_id = _offline_template(prompt), None
    else:
        proposed, response_id = _meta_proposal(prompt, client)
    normalized, world = validate_proposal(proposed)
    canonical = json.dumps(asdict(world), sort_keys=True, allow_nan=False)
    return {
        "schema": "emergent-world-design-v1",
        "source": source,
        "model": META_MODEL if source == "meta" else None,
        "meta_response_id": response_id,
        "challenge_id": hashlib.sha256(canonical.encode()).hexdigest()[:16],
        "prompt": prompt.strip(),
        "proposal": normalized,
        "world": asdict(world),
        "limitations": (
            "Meta designs world parameters only; it does not direct cell policies."
            if source == "meta" else
            "Deterministic keyword-based local template; NO Meta API request happened."
        ),
    }


def evaluate_world(design: dict, seeds: list[int] | None = None,
                   steps: int = 90, count: int = 64) -> dict:
    if not isinstance(design, dict) or "proposal" not in design:
        raise DesignError("A previously validated world design is required")
    _, world = validate_proposal(design["proposal"])
    if seeds is None:
        seeds = [101, 102, 103]
    if not isinstance(seeds, list) or not (1 <= len(seeds) <= 12):
        raise DesignError("Need 1..12 unique seeds")
    if any(type(seed) is not int or seed < 0 or seed > 2**32 - 1 for seed in seeds):
        raise DesignError("Seeds must be nonnegative 32-bit integers")
    if len(set(seeds)) != len(seeds):
        raise DesignError("All seeds must be distinct")
    if type(steps) is not int or not 1 <= steps <= 500:
        raise DesignError("steps must be 1..500")
    if type(count) is not int or not 1 <= count <= 256:
        raise DesignError("cells must be 1..256")
    modes = ("random", "frozen", "greedy", "learned", "learned_no_messages")
    rows = []
    for seed in seeds:
        for variant in modes:
            silent = variant == "learned_no_messages"
            mode = "learned" if silent else variant
            spec = RunSpec(
                seed=seed, steps=steps, mode=mode, world=world,
                cells=CellSpec(count=count, communication_radius=0 if silent else 3),
            )
            outcome = run(spec)
            rows.append({
                "seed": seed, "variant": variant,
                "survival_fraction": outcome["survival_fraction"],
                "remaining_energy": outcome["remaining_energy"],
                "harvested_total": outcome["harvested_total"],
                "alive_cell_steps": outcome["alive_cell_steps"],
                "messages": outcome["neighbor_messages"],
                "steps_completed": outcome["steps_completed"],
            })
    aggregated = {}
    for variant in modes:
        batch = [r for r in rows if r["variant"] == variant]
        aggregated[variant] = {
            "survival_mean": sum(r["survival_fraction"] for r in batch) / len(batch),
            "energy_mean": sum(r["remaining_energy"] for r in batch) / len(batch),
            "harvested_mean": sum(r["harvested_total"] for r in batch) / len(batch),
        }
    return {
        "schema": "emergent-world-evaluation-v1",
        "challenge_id": design["challenge_id"],
        "designer_source": design["source"],
        "model": design["model"],
        "seeds": seeds,
        "steps": steps,
        "cells": count,
        "aggregate": aggregated,
        "rows": rows,
        "limitations": [
            "Exploratory development controls; no held-out preregistered tests.",
            "Resource worlds do not guarantee cooperation is necessary.",
            "Energy is simulated, not joules; no physical efficiency demonstrated.",
            "Greedy is a local hand-coded baseline; it may outperform learning.",
            "If source=template, NO real Meta model was invoked.",
        ],
    }
