"""Live HTTP smoke test. Run while Studio is listening on localhost:8765."""
import json
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:8765"


def call(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(BASE + path, data=data,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=45) as reply:
        raw = reply.read()
    return json.loads(raw) if path != "/" else raw.decode("utf-8")


def main():
    health = call("/api/health")
    assert health["ok"] and health["local_only"], health
    page = call("/")
    assert "Counterfactual evidence" in page
    assert "Meta Muse Spark" in page
    assert 'id="download-evidence"' in page

    baseline = call("/api/status")
    try:
        before = call("/api/reset", {"seed": 13001, "changed": False,
                                      "stochastic": False,
                                      "controller": "no_messages"})
        assert before["controller"] == "no_messages"
        compare = call("/api/evidence", {"seed": 13001, "trials": 4,
                                          "changed": False})
        assert len(compare["records"]) == 4
        assert call("/api/status") == before, "Comparison changed the mission"
        after = call("/api/step", {"count": 20})
        assert after["steps"] == 20
        assert after["messages"] == 0
        fault = call("/api/fault", {})
        assert fault["faults_injected"] == 1
        chat = call("/api/chat", {"message": "status", "provider": "local"})
        assert "Mission status:" in chat["answer"]
        if not after["meta_configured"]:
            remote = call("/api/chat", {
                "message": "Explain the science", "provider": "meta"})
            assert "not configured" in remote["answer"], remote
        print(json.dumps({
            "health": "OK",
            "html": "OK",
            "paired_records": len(compare["records"]),
            "event_completion": compare["summary"]["event"]["completed"],
            "queued_messages_disabled": after["messages"] == 0,
            "memory_failure": fault["faults_injected"],
            "meta_configured": after["meta_configured"],
            "mission_isolation": "OK",
        }, indent=2))
    finally:
        call("/api/reset", {"seed": baseline["seed"],
                             "changed": baseline["changed"],
                             "stochastic": baseline["stochastic"],
                             "controller": baseline["controller"]})


if __name__ == "__main__":
    main()
