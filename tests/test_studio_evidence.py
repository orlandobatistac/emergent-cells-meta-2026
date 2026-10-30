"""Scientific controls and local browser comparison tests."""
import unittest

from emergent_cells.experiments.studio import Laboratory
from emergent_cells.experiments.studio_evidence import compare_controllers


class PairedEvidenceTests(unittest.TestCase):
    def test_paired_reproducibility_and_message_ablation(self):
        a = compare_controllers(start_seed=13001, trials=2, changed=False)
        b = compare_controllers(start_seed=13001, trials=2, changed=False)
        self.assertEqual(a["records"], b["records"])
        self.assertEqual(a["summary"], b["summary"])
        self.assertEqual(len(a["records"]), 2)
        for record in a["records"]:
            self.assertEqual(set(record["outcomes"]),
                             {"event", "no_messages", "synchronous"})
            self.assertEqual(record["outcomes"]["no_messages"]["messages"], 0)
        self.assertTrue(a["protocol"]["authored_policy_and_messages"])
        self.assertTrue(a["protocol"]["exploratory_not_preregistered"])

    def test_invalid_requests(self):
        for kwargs in (
            {"start_seed": -1}, {"start_seed": True},
            {"start_seed": 10_000_000, "trials": 2},
            {"start_seed": 100, "trials": 13},
            {"start_seed": 100, "trials": 0},
            {"start_seed": 100, "changed": 1},
        ):
            with self.assertRaises(ValueError):
                compare_controllers(**kwargs)

    def test_live_ablation_mode(self):
        lab = Laboratory()
        status = lab.reset(seed=13001, changed=False, stochastic=False,
                           controller="no_messages")
        self.assertEqual(status["controller"], "no_messages")
        self.assertIn("without queued messages", status["policy"])
        status = lab.step(20)
        self.assertEqual(status["messages"], 0)

    def test_http_evidence_preserves_active_mission(self):
        from fastapi.testclient import TestClient
        from emergent_cells.experiments.studio import create_app
        lab = Laboratory()
        client = TestClient(create_app(lab))
        before = lab.status()
        result = client.post("/api/evidence", json={"seed": 13001, "trials": 2})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(len(result.json()["records"]), 2)
        self.assertEqual(lab.status(), before)
        self.assertEqual(client.post("/api/evidence", json={
            "seed": 13001, "trials": 13}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
