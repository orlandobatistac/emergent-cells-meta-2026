# Sequential transfer through local reward-conditioned policy recombination — 2026-10-28

## Research question

Can a population of simulated local-reward learners reuse parts of previously learned policies to accelerate learning of an unfamiliar rule without overwriting those source memories? We test **OR** and **NAND**, neither of which was used to train the three source banks (**XOR, XNOR, AND**).

This is a controlled, pairwise, four-input Boolean simulation. It does **not** demonstrate general intelligence, autonomous task discovery, physical energy savings, substrate independence, unbounded growth, or a general-purpose continual-learning algorithm.

## Relevant research (inspiration, not reproduction)

- Barry, Gerstner & Bellec, *Context selectivity with dynamic availability enables lifelong continual learning*, Neural Networks, 2025. https://www.sciencedirect.com/science/article/pii/S0893608025006082
- *Bayesian continual learning and forgetting in neural networks*, Nature Communications, 2025 (MESU). https://www.nature.com/articles/s41467-025-64601-w
- Chen et al., *Parameter-efficient Continual Learning for Enhancing Plasticity without Forgetting under Limited Model Capacity*, CVPR 2026 (GRAPA). https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Parameter-efficient_Continual_Learning_for_Enhancing_Plasticity_without_Forgetting_under_Limited_CVPR_2026_paper.html
- Le et al., *Task-Free Continual Learning via Order-Invariant Linearized Adaptation and Density-Guided Adapter Routing*, UAI 2026 (HESTIA). https://proceedings.mlr.press/v337/le26a.html
- *Building adaptive knowledge bases for evolving continual learning models*, npj Artificial Intelligence, 2025 (CABLE). https://www.nature.com/articles/s44387-025-00028-4

## Setup and restrictions

1. For each seed, 24 fixed pairs (48 independent local learners) pretrain one learned binary peer-message protocol and three local response banks, one per source Boolean gate, for 3,800 interactions per source. **The experiment scheduler uses explicit bank/task IDs for source pretraining**; it is not task-free during that phase. Sender weights are frozen after the first phase.
2. For a new OR or NAND world, the experiment scheduler probes each frozen source bank for 120 interactions (360 probing interactions total). Each local learner sees its private cue, its peer's learned bit and local received reward. No correct target, world rule ID, or reward-availability bit is given to the learner. Probe bank scheduling is authored, not autonomously learned.
3. **Slot mosaic** estimates reward per local input slot (own bit + received learned symbol) and source bank. It uses a Beta(1,1) smoothing score `(local_hits+1)/(local_visits+2)`; each cell chooses a source-policy logit for each of its four slots. The assembled new bank's logits are multiplied by **0.55** to mitigate overconfidence and permit further local reward-driven learning. This is an engineered transfer heuristic, not a learned gating policy.
4. Controls: cold-start bank; cold-start bank with **360 additional local training interactions** to match the probe budget; copy last bank; copy highest-reward bank; softened highest-reward-bank copy (0.35); and copy the worst-performing bank as an intentionally adverse transfer control. All variants train the novel bank with the same 1,100 interaction stream, using independent reward-modulated cell updates; no correct-action labels. Old source banks are kept frozen by experiment design.
5. Outcomes use **both cells in a pair being correct** as a success, independent of reward dropout. Tail accuracy is over the last 200 interactions; final accuracy uses a separate frozen-policy evaluation stream of 300 interactions. The seed and current rule are accessible to simulation code to generate rewards and evaluation targets, not to novel-policy learners.

## Exploratory study

Seeds 11–12, 16 pairs, 3,200 pretraining interactions per source, 900 novel-training interactions. The new slot mosaic method obtained **90.65% OR** and **90.80% NAND** post-adaptation evaluation accuracy, versus **81.48%** and **81.44%** from cold start. This small pilot was used to select the 0.55 mosaic-logit scaling; these numbers are developmental evidence, not a confirmatory estimate. The stronger equal-budget baseline was added after the exploratory run.

## Held-out validation, fresh seeds 2001–2005

24 pairs per seed, 3,800 interactions per source, 120 probes for each old bank, 1,100 novel-task interactions, simulated reward dropout 8%.

| Novel rule / variant | First 200 training interactions | Final separate-test accuracy |
|---|---:|---:|
| OR — cold start | 32.83% | 85.49% |
| OR — cold start + 360 extra training interactions | 63.48% | 89.20% |
| OR — full copy of best source | 69.81% | 71.13% |
| **OR — slot mosaic transfer** | **83.58%** | **92.04%** |
| NAND — cold start | 32.45% | 85.26% |
| NAND — cold start + 360 extra training interactions | 63.52% | 89.00% |
| NAND — full copy of best source | 69.74% | 70.54% |
| **NAND — slot mosaic transfer** | **83.06%** | **91.73%** |

Relative to the stronger extra-training control, mosaic yields **+20.10 percentage points (OR)** and **+19.54 points (NAND)** on the first 200 novel-task steps, and **+2.84 points (OR)** and **+2.73 points (NAND)** on final separate-test accuracy, averaged across five independent seeds.

Raw per-seed data, including all seven variants and all source-probe outcomes:
`results/audited/2026-10-28-sequential-transfer-holdout.json`.

The results show **faster local adaptation in these synthetic rules**. Whole-bank copying shows negative transfer: it can preserve an overconfident incorrect choice. Sparse local recombination reuses correct prior input-conditioned responses while allowing additional learning.

## Additional exploratory high-noise stress

With a different three-seed exploratory schedule (seeds 301–303, 16 pairs, 3,200 pretraining steps, 900 novel adaptation steps), simulated reward dropout was increased from 8% to **20%** for both source training and adaptation. Slot mosaic achieved **88.73% OR** and **88.01% NAND** final separate-test accuracy, versus **85.08% OR** and **84.47% NAND** from the equal-interaction-budget cold-start control. Its first-200-step accuracy was **78.64% OR** and **77.00% NAND**, versus **60.13%** and **59.36%** for the extra-training control.

This is exploratory and changes pairs/training lengths as well as dropout; it does **not** isolate the dropout effect or demonstrate robust context-change detection. It does indicate that the transfer advantage did not vanish outright in this selected small-sample stress. Raw file: `results/exploratory/sequential-transfer-noise-20pct.json`.

## Limitations and next falsifiable experiment

- OR/NAND share exactly the same two-bit input format and output action space as the three learned sources. This is **same-family recombination**, not generalization to unrelated or compositional real-world tasks.
- All 16 two-input Boolean functions can in principle be composed from source banks with sufficiently diverse logits; the task family may be too small to establish meaningful general transfer.
- Each memory slot is pre-enumerated by a designer. Probe scheduling, memory sizing, transfer choice and confidence scale are engineered. Task boundary and pretrained source IDs are explicit in the experimental scheduler. This is not fully autonomous task-free learning.
- Each variant has a different computational/probe cost. The stronger cold-start control matches **interaction count**, not probe/switch memory allocation, update operations, wall-clock speed or measured electricity.
- Frozen source banks retain earlier weights by construction. We have **not** demonstrated autonomous retrieval of old tasks after sequentially learning new ones, or protected plasticity under shared resource constraints.
- Five seeds and two related rule types are too few for broad significance or external-validity claims.
- Next: sequential multi-step contexts, reused building blocks under changed observation mappings, unannounced shifts, shared constrained memory, and local routing **without scheduler-provided task IDs**. Measure forward transfer (FWT), backward transfer (BWT), forgetting and joules/CPU timing separately.

## Reproduce locally

```powershell
python -m pip install -e .
python -m unittest discover -s tests -q
python run.py sequential-transfer --seeds 2001,2002,2003,2004,2005 --pairs 24 --pre-steps 3800 --probe-steps 120 --adapt-steps 1100 --output results/runs/transfer-reproduction.json
```

The experiment is Python/NumPy only; it does not call Meta APIs. Existing historical data files are not rewritten.
