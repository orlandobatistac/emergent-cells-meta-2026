"""Mocked Meta integration: no credentials or API calls needed in CI."""
import io
import json
import os
import unittest
from unittest.mock import patch

from emergent_cells.experiments.studio_meta import (
    ask_meta_model, is_configured, ENDPOINT,
)


class MetaModelTests(unittest.TestCase):
    def test_absent_key_keeps_offline_lab_available(self):
        with patch.dict(os.environ, {"MODEL_API_KEY": ""}):
            self.assertFalse(is_configured())
            self.assertIn("not configured",
                          ask_meta_model("Explain", {"steps": 4}))

    def test_mocked_response_and_no_hidden_map(self):
        class FakeResponse(io.BytesIO):
            pass

        payload = {"choices": [{"message": {
            "content": "The network uses authored local values."}}]}
        captured = []

        def fake_open(req, timeout):
            captured.append(req)
            return FakeResponse(json.dumps(payload).encode("utf-8"))

        with patch.dict(os.environ, {"MODEL_API_KEY": "fake-private-key"}):
            with patch("emergent_cells.experiments.studio_meta.request.urlopen",
                       side_effect=fake_open):
                answer = ask_meta_model("Summarize", {"steps": 9}, timeout=2)
        self.assertIn("authored local values", answer)
        self.assertEqual(captured[0].full_url, ENDPOINT)
        sent = json.loads(captured[0].data)
        self.assertEqual(sent["model"], "muse-spark-1.3")
        self.assertNotIn("true_position", str(sent))
        self.assertNotIn("fake-private-key", str(sent))

    def test_http_provider_is_opt_in(self):
        from fastapi.testclient import TestClient
        from emergent_cells.experiments.studio import create_app
        with patch.dict(os.environ, {"MODEL_API_KEY": ""}):
            client = TestClient(create_app())
            response = client.post("/api/chat", json={
                "message": "Explain cellular intelligence",
                "provider": "meta",
            })
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn("not configured", response.json()["answer"])
            self.assertEqual(client.post("/api/chat", json={
                "message": "Explain", "provider": "bad",
            }).status_code, 422)


if __name__ == "__main__":
    unittest.main()
