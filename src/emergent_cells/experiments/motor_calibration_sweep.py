"""Exploratory calibration sample-efficiency curves (motor permutations only)."""
import json
from pathlib import Path
import random
from emergent_cells.experiments.learned_motor_code import PERMUTATIONS, learn_mapping


def main():
    output = {}
    seeds = list(range(601, 651))
    for noise in (0., .08, .20):
        by_budget = {}
        for steps in (12, 24, 48, 72, 100, 140):
            scores = []
            confidence = []
            for seed in seeds:
                mapping = PERMUTATIONS[random.Random(seed * 433 + 19).randrange(24)]
                learner = learn_mapping(seed, mapping,
                                        calibration_steps=steps, noise=noise)
                scores.append(learner.estimate() == mapping)
                confidence.append(learner.posterior_confidence())
            by_budget[str(steps)] = {
                "accuracy": sum(scores)/len(scores),
                "mean_confidence": sum(confidence)/len(confidence),
                "count": len(scores)}
        output[str(noise)] = by_budget
    payload = {"protocol": {
        "development_seeds": seeds,
        "test": "exact command-permutation recovery from local wall-contact pairs",
        "motor_hypothesis_count": 24,
        "independent_source_calibration_actions_per_seed": True,
        "note": "dev-only; independent holdout uses 9101-9200 / 9301-9350"
        }, "summary": output}
    p=Path("results/exploratory/2026-10-29-motor-calibration-sample-efficiency.json")
    p.parent.mkdir(exist_ok=True,parents=True)
    p.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(output,indent=2))


if __name__ == "__main__":
    main()
