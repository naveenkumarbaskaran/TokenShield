from tokenshield.shield import Shield
from tokenshield.router import CostRouter
from tokenshield.replay import CostReplay


def _call(shield, content="hello"):
    return shield.call(messages=[{"role": "user", "content": content}])


def test_replay_log_records_calls():
    shield = Shield(model="gpt-4o")
    _call(shield, "a")
    _call(shield, "b")
    assert len(shield._replay_log) == 2


def test_replay_log_entry_structure():
    shield = Shield(model="gpt-4o")
    _call(shield, "test")
    entry = shield._replay_log[0]
    assert "messages" in entry
    assert "tools" in entry
    assert "model_used" in entry
    assert "input_tokens" in entry
    assert "output_tokens" in entry
    assert "cost" in entry


def test_replay_with_cheaper_router_shows_savings():
    shield = Shield(model="gpt-4o")
    for _ in range(3):
        _call(shield, "hi")  # simple → routes to gpt-4o at full price

    cheaper_router = CostRouter(simple="gpt-4o-mini", medium="gpt-4o", complex="gpt-4o")
    report = shield.replay(router=cheaper_router)

    assert report["original_cost"] > 0
    assert report["replayed_cost"] < report["original_cost"]
    assert report["savings"] == round(report["original_cost"] - report["replayed_cost"], 6)
    assert 0 < report["savings_pct"] <= 100
    assert report["call_count"] == 3


def test_replay_same_config_zero_savings():
    shield = Shield(model="gpt-4o")
    _call(shield, "hi")
    report = shield.replay()  # no router override — same model
    assert report["savings"] == 0.0
    assert report["savings_pct"] == 0.0


def test_replay_empty_log():
    shield = Shield(model="gpt-4o")
    report = shield.replay()
    assert report["call_count"] == 0
    assert report["original_cost"] == 0.0
    assert report["replayed_cost"] == 0.0


def test_replay_does_not_modify_log():
    shield = Shield(model="gpt-4o")
    _call(shield, "x")
    _call(shield, "y")
    shield.replay()
    assert len(shield._replay_log) == 2


def test_cost_replay_standalone():
    shield = Shield(model="gpt-4o")
    for _ in range(5):
        _call(shield, "hi")
    replay = CostReplay(shield)
    report = replay.run(router=CostRouter(simple="gpt-4o-mini", medium="gpt-4o", complex="gpt-4o"))
    assert report["call_count"] == 5
    assert "savings_pct" in report


def test_replay_with_model_override():
    shield = Shield(model="gpt-4o")
    for _ in range(3):
        _call(shield, "hi")
    report = shield.replay(model="gpt-4o-mini")
    assert report["call_count"] == 3
    assert report["replayed_cost"] < report["original_cost"]
    assert report["per_model"] == {"gpt-4o-mini": 3}
    assert report["savings"] > 0
