"""Train cell-private 3-action policies: EXPLOIT, PROBE, or REST.

Research task only. Cells never see world scenario names, target actions, or
resource availability. A deliberately authored Bayesian belief, action slots,
and coordinated probe primitive are retained from earlier research.

REST is an actual no-lever/no-resource action that avoids synthetic activity
cost. The whole pair's participation outcome is environmentally observable;
it is NOT a privileged resource availability sensor. A cell that chooses REST
cannot obtain a new reward sample and therefore cannot update its belief.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .active_probes import ProbeSpec, pretrain, sigmoid
from .learned_curiosity import (
    SCENARIOS, HOLDOUT_SCENARIOS, ALL_SCENARIOS,
    local_state, update_belief,
)

# A=exploitation, B=expensive counterfactual test, C=skip active expenditure.
EXPLOIT, PROBE, REST = range(3)
TEST_SCENARIOS = HOLDOUT_SCENARIOS + (
    "severe_drought", "global_change_blackout",
)
SCENARIO_NAMES = SCENARIOS + TEST_SCENARIOS + ("training_sparse_pulses",)


@dataclass(frozen=True)
class ThreeActionSpec:
    pairs: int = 24
    pretrain: int = 2400
    train_episodes: int = 140
    train_steps: int = 55
    eval_steps: int = 300
    alpha: float = .09
    epsilon_start: float = .26
    epsilon_end: float = .05
    active_cost: float = .12
    probe_cost: float = .025
    message_cost: float = .004
    intrinsic_weight: float = .20
    pulse_rate: float = .92

    def validate(self):
        if not 4 <= self.pairs <= 2000:
            raise ValueError("pairs out of range")
        if not 100 <= self.pretrain <= 100000:
            raise ValueError("pretrain out of range")
        if not 20 <= self.train_episodes <= 2000:
            raise ValueError("episodes out of range")
        if not 15 <= self.train_steps <= 1000 or not 100 <= self.eval_steps <= 5000:
            raise ValueError("episode length out of range")
        if not 0 < self.alpha <= .9 or not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("learning settings invalid")
        if not 0 <= self.active_cost <= 1 or not 0 <= self.probe_cost <= .4:
            raise ValueError("activity costs invalid")
        if not 0 <= self.message_cost <= .4 or not 0 <= self.intrinsic_weight <= 2:
            raise ValueError("other cost or progress bonus invalid")
        if not .3 < self.pulse_rate < 1:
            raise ValueError("pulse availability invalid")


class ThreeActionCells:
    def __init__(self, spec: ThreeActionSpec, seed: int, *,
                 mode: str = "learned_three"):
        spec.validate()
        if mode not in ("learned_three", "learned_two",
                        "learned_three_no_history",
                        "learned_three_low_supply", "learned_two_low_supply",
                        "learned_three_ema", "learned_two_ema",
                        "manual_two",
                        "guard_two", "always_rest"):
            raise ValueError("unrecognized mode")
        self.spec = spec
        self.seed = seed
        self.mode = mode
        self.n = spec.pairs * 2
        self.ids = np.arange(self.n)
        self.peer = self.ids ^ 1
        # Both learned arms own identical-size tables. Third action is masked
        # in learned_two, not a smaller architecture.
        self.q = np.zeros((self.n, 288 if mode.endswith("_ema") else 96, 3),
                          dtype=float)
        self.q[:, :, EXPLOIT] = .25
        self.q[:, :, PROBE] = .20
        self.q[:, :, REST] = -.05
        self.reset()

    def reset(self):
        self.belief = np.broadcast_to(
            np.array([.90, .05, .05]), (self.n, 4, 3)
        ).copy()
        self.corrected = np.zeros((self.n, 4), dtype=bool)
        self.failed = np.zeros((self.n, 4), dtype=np.int16)
        # This is factual LOCAL history: missing reward on own attempted
        # active interactions; NOT a scenario label or resource sensor.
        self.loss_streak = np.zeros((self.n, 4), dtype=np.int16)
        self.global_reward_rate = np.full(self.n, .75, dtype=float)

    @staticmethod
    def world(bits, scenarios, pulse_rate):
        cue_pair = bits[::2]*2 + bits[1::2]
        goal = bits[::2] ^ bits[1::2]
        flip = (
            np.isin(scenarios, ["global_change", "global_change_drought",
                                  "global_change_blackout"])
            | ((scenarios == "changed_11") & (cue_pair == 3))
            | ((scenarios == "changed_01") & (cue_pair == 1))
            | (np.isin(scenarios, ["changed_10", "changed_10_drought",
                                    "mixed_change_10_scarcity_01"]) &
               (cue_pair == 2))
        )
        goal = goal ^ flip.astype(int)
        available = np.full(len(cue_pair), pulse_rate)
        available = np.where(
            np.isin(scenarios, ["uniform_drought", "changed_10_drought",
                                 "global_change_drought"]), .55, available)
        available = np.where(
            np.isin(scenarios, ["severe_drought", "global_change_blackout"]),
            .05, available)
        available = np.where(scenarios == "training_sparse_pulses",
                             .08, available)
        available = np.where(
            ((scenarios == "scarcity_11") & (cue_pair == 3))
            | ((scenarios == "scarcity_10") & (cue_pair == 2))
            | (np.isin(scenarios, ["scarcity_01",
                                    "mixed_change_10_scarcity_01"]) &
               (cue_pair == 1)), 0., available)
        available = np.where(
            (scenarios == "scarcity_01_partial") & (cue_pair == 1),
            .20, available)
        return goal, available, cue_pair

    def select(self, state, rng, *, train, epsilon):
        if self.mode == "always_rest":
            return np.full(self.n, REST, dtype=np.int64)
        if self.mode in ("manual_two", "guard_two"):
            request = np.isin(state // 16, [1, 2, 5])
            if self.mode == "guard_two":
                request &= self.failed[self.ids, self._slot] < 2
            return np.where(request, PROBE, EXPLOIT).astype(np.int64)

        choices = self.q[self.ids, state]
        if self.mode.startswith("learned_two"):
            choices = choices.copy()
            choices[:, REST] = -np.inf
        decision = choices.argmax(axis=1)
        if train:
            explore = rng.random(self.n) < epsilon
            if self.mode.startswith("learned_two"):
                samples = rng.integers(0, 2, self.n)
            else:
                samples = rng.integers(0, 3, self.n)
            decision = np.where(explore, samples, decision)
        return decision.astype(np.int64)

    def run(self, send, recv, scenarios, rng, *,
            steps: int, train: bool, epsilon: float):
        if len(scenarios) != self.spec.pairs or any(
                s not in SCENARIO_NAMES for s in scenarios):
            raise ValueError("invalid world assignment")
        self.reset()
        scenarios = np.array(scenarios)
        ix, peer, spec = self.ids, self.peer, self.spec
        correct = harvested = cost = 0.
        rest_events = probes = messages = 0
        case_hit = np.zeros(4, dtype=int)
        case_seen = np.zeros(4, dtype=int)
        for _ in range(steps):
            bits = rng.integers(0, 2, self.n)
            symbol = (sigmoid(send[:,0] + send[:,1]*(bits*2-1)) >= .5)
            slot = bits*2 + symbol[peer].astype(int)
            self._slot = slot  # each cell observes its own private slot
            belief = self.belief[ix, slot]
            base = local_state(belief)
            state = (base*16 + np.minimum(self.failed[ix, slot], 3)*4 +
                     np.minimum(self.loss_streak[ix, slot], 3))
            if self.mode == "learned_three_no_history":
                state = base*16 + np.minimum(self.failed[ix, slot], 3)*4
            if self.mode.endswith("_ema"):
                # Author-engineered COARSE FEATURE of own experience,
                # not an availability sensor or a task label.
                level = np.where(self.global_reward_rate < .25, 0,
                                 np.where(self.global_reward_rate < .65, 1, 2))
                state = state*3 + level
            intent = self.select(state, rng, train=train, epsilon=epsilon)
            # Fixed local correction suppresses further probes on known slots.
            intent = np.where(
                self.corrected[ix,slot] & (intent == PROBE),
                EXPLOIT, intent
            )
            active = intent != REST
            both_active = active[::2] & active[1::2]
            request = intent == PROBE
            probe_pair = (request[::2] | request[1::2]) & both_active
            coordinated = np.repeat(probe_pair, 2)
            base_act = (sigmoid(recv[ix,slot]) >= .5).astype(int)
            invert = self.corrected[ix,slot] ^ coordinated
            actions = base_act ^ invert.astype(int)
            goal, available, case = self.world(bits, scenarios, spec.pulse_rate)
            opened = (actions[::2] == goal) & (actions[1::2] == goal) & both_active
            payout = opened & (rng.random(spec.pairs) < available)
            reward = np.repeat(payout.astype(float), 2)
            net = (
                reward - spec.active_cost*active.astype(float)
                - spec.probe_cost*coordinated.astype(float)
                - spec.message_cost*request.astype(float)
            )
            informed = np.repeat(both_active,2)
            prior = belief.copy()
            posterior = update_belief(
                prior, reward.astype(int), invert, spec.pulse_rate)
            posterior = np.where(informed[:,None], posterior, prior)
            if train and self.mode.startswith("learned"):
                # Q values learn exclusively from private net reward and
                # optionally local posterior progress, NEVER world goal.
                h_before = -np.sum(prior*np.log2(np.maximum(prior,1e-12)),axis=1)
                h_after = -np.sum(
                    posterior*np.log2(np.maximum(posterior,1e-12)),axis=1)
                progress = np.maximum(0.,h_before-h_after)
                bonus = spec.intrinsic_weight*progress*(intent == PROBE)
                value = self.q[ix,state,intent]
                # Peer-initiated probe/rest changes own outcome. Suppress
                # confounded Q updates unless both partners choose
                # compatible actions. REST gets its own observable 0 payoff.
                independent = ~(request[peer] & ~request)
                independent &= ~((intent != REST) & (intent[peer] == REST))
                gradient = spec.alpha*(net + bonus - value)
                self.q[ix[independent],state[independent],
                       intent[independent]] += gradient[independent]
                np.clip(self.q,-2.,2.,out=self.q)
            self.belief[ix,slot] = posterior
            failed_probe = coordinated & informed & (reward == 0)
            succeeded_probe = coordinated & informed & (reward > 0)
            self.failed[ix,slot] = np.where(
                failed_probe,
                np.minimum(10,self.failed[ix,slot]+1),
                np.where(succeeded_probe,0,self.failed[ix,slot])
            )
            self.corrected[ix,slot] |= posterior[:,2] > .96
            # A cell sees only its OWN received reward in active periods.
            self.global_reward_rate = np.where(
                informed, .985*self.global_reward_rate + .015*reward,
                self.global_reward_rate
            )
            # REST supplies no reward evidence and does not alter
            # this action-outcome history.
            self.loss_streak[ix,slot] = np.where(
                informed & (reward <= 0),
                np.minimum(10,self.loss_streak[ix,slot]+1),
                np.where(informed & (reward > 0),0,
                         self.loss_streak[ix,slot])
            )
            correct += float(opened.sum())
            harvested += float(payout.sum())
            cost += float((spec.active_cost*active.sum()/2
                           +spec.probe_cost*probe_pair.sum()
                           +spec.message_cost*request.sum()/2))
            rest_events += int((intent == REST).sum())
            probes += int(probe_pair.sum())
            messages += int(request.sum())
            case_seen += np.bincount(case, minlength=4)
            case_hit += np.bincount(
                case, weights=opened.astype(int), minlength=4).astype(int)
        denom = steps * spec.pairs
        return {
            "correct": correct/denom,
            "pulses": harvested/denom,
            "net_energy": (harvested-cost)/denom,
            "rest_fraction": rest_events/(2*denom),
            "requests": messages,
            "probes": probes,
            "synthetic_cost": cost,
            "per_case": {
                format(i,"02b"): float(case_hit[i]/max(1,case_seen[i]))
                for i in range(4)
            },
        }


def train(seed: int, spec: ThreeActionSpec, mode: str, pretrained):
    learner = ThreeActionCells(spec, seed, mode=mode)
    if not mode.startswith("learned"):
        return learner
    if mode.endswith("_low_supply") or mode.endswith("_ema"):
        # The learner NEVER receives this label: world-only rate change.
        # The 0.08 pulse regime is TRAINING, the 0.05 blackout remains unseen.
        categories = np.array(SCENARIOS + ("training_sparse_pulses",))
        frequencies = np.array([.12,.135,.135,.105,.105,.075,.075,.25])
    else:
        categories = np.array(SCENARIOS)
        frequencies = np.array([.16,.18,.18,.14,.14,.10,.10])
    rng = np.random.default_rng(seed + 3141590)
    for ep in range(spec.train_episodes):
        scenes = rng.choice(categories, size=spec.pairs, p=frequencies)
        epsilon = (spec.epsilon_start +
                   (spec.epsilon_end-spec.epsilon_start)*ep /
                   max(1,spec.train_episodes-1))
        learner.run(pretrained[0], pretrained[1], scenes, rng,
                    steps=spec.train_steps, train=True, epsilon=epsilon)
    return learner


def benchmark(seeds, spec=ThreeActionSpec()):
    spec.validate()
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("nonempty unique seeds")
    methods = ("learned_three", "learned_two",
               "learned_three_no_history",
               "learned_three_low_supply", "learned_two_low_supply",
               "learned_three_ema", "learned_two_ema",
               "manual_two",
               "guard_two", "always_rest")
    result = []
    summary = {}
    for seed in seeds:
        pre = pretrain(seed, ProbeSpec(
            pairs=spec.pairs, pretrain=spec.pretrain,
            horizon=spec.eval_steps
        ))
        learners = {mode: train(seed, spec, mode, pre) for mode in methods}
        for scenario in TEST_SCENARIOS:
            for mode in methods:
                rng = np.random.default_rng(
                    seed+79813+SCENARIO_NAMES.index(scenario)*100003
                )
                original = learners[mode].q.copy()
                row = learners[mode].run(
                    pre[0], pre[1], [scenario]*spec.pairs, rng,
                    steps=spec.eval_steps, train=False, epsilon=0)
                if not np.array_equal(original,learners[mode].q):
                    raise AssertionError("eval changed learned Q")
                result.append({
                    **row, "seed":seed, "scenario":scenario,"method":mode})
    fields = ("correct", "pulses", "net_energy", "rest_fraction",
              "requests", "probes", "synthetic_cost")
    for scenario in TEST_SCENARIOS:
        summary[scenario] = {}
        for mode in methods:
            items = [r for r in result if r["scenario"]==scenario
                     and r["method"]==mode]
            summary[scenario][mode] = {
                field:float(np.mean([r[field] for r in items]))
                for field in fields
            }
            summary[scenario][mode]["per_case"] = {
                c:float(np.mean([r["per_case"][c] for r in items]))
                for c in ("00","01","10","11")
            }
    return {
        "schema":"private-three-action-learning-v1",
        "seeds":list(seeds),"spec":asdict(spec),"methods":list(methods),
        "train_scenarios":list(SCENARIOS),
        "low_supply_training_extra_world": "training_sparse_pulses (pulse 0.08; world-only)",
        "held_out":list(TEST_SCENARIOS),
        "summary":summary,"rows":result,
        "limitations":[
            "REST action and its activity cost are engineered, not discovered.",
            "Paired gate participation is observable; private resource availability is not.",
            "Manual Bayesian state classes and action inversion remain engineered.",
            "Non-stationary worlds drawn from the familiar XOR-flip-scarcity family.",
            "3-action Q and 2-action Q have same parameter size; last action masked in 2.",
            "Net costs are symbolic; physical joules and per-cell RAM not measured.",
        ],
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds",default="11,12")
    p.add_argument("--pairs",type=int,default=16)
    p.add_argument("--pretrain",type=int,default=1600)
    p.add_argument("--train-episodes",type=int,default=90)
    p.add_argument("--train-steps",type=int,default=40)
    p.add_argument("--eval-steps",type=int,default=180)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    data=benchmark([int(t) for t in a.seeds.split(",") if t.strip()],
                   ThreeActionSpec(pairs=a.pairs,pretrain=a.pretrain,
                       train_episodes=a.train_episodes,
                       train_steps=a.train_steps,eval_steps=a.eval_steps))
    body=json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(body,encoding="utf8")
        print("SAVED",a.output)
    else:
        print(body)


if __name__=="__main__":
    main()
