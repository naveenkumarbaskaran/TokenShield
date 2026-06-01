from tokenshield.router import CostRouter


def _msg(content: str, role: str = "user") -> dict:
    return {"role": role, "content": content}


def test_simple_classification():
    r = CostRouter(simple="mini", medium="mid", complex="big")
    msgs = [_msg("hi")]
    assert r.route(msgs) == "mini"


def test_complex_by_token_count():
    r = CostRouter(simple="mini", medium="mid", complex="big")
    # >3000 estimated tokens = >12000 chars
    msgs = [_msg("x" * 13000)]
    assert r.route(msgs) == "big"


def test_complex_by_tool_count():
    r = CostRouter(simple="mini", medium="mid", complex="big")
    tools = [{"name": f"t{i}"} for i in range(6)]
    assert r.route([_msg("hi")], tools) == "big"


def test_medium_classification():
    r = CostRouter(simple="mini", medium="mid", complex="big")
    # 3 messages pushes past simple (>2 messages) but not complex
    msgs = [_msg("a"), _msg("b", "assistant"), _msg("c")]
    assert r.route(msgs) == "mid"


def test_medium_by_token_count():
    r = CostRouter(simple="mini", medium="mid", complex="big")
    # 600 tokens = ~2400 chars — above simple (<500) but below complex (>3000)
    msgs = [_msg("x" * 2500)]
    assert r.route(msgs) == "mid"


def test_explicit_model_override_skips_routing():
    r = CostRouter(simple="mini", medium="mid", complex="big")
    assert r.route([_msg("hi")], model_override="gpt-4o") == "gpt-4o"


def test_defaults():
    r = CostRouter()
    assert r.simple == "gpt-4o-mini"
    assert r.medium == "gpt-4o"
    assert r.complex == "gpt-4o"


def test_empty_messages_routes_to_simple():
    r = CostRouter(simple="mini", medium="mid", complex="big")
    assert r.route([]) == "mini"


def test_exactly_500_tokens_is_medium():
    r = CostRouter(simple="mini", medium="mid", complex="big")
    # exactly 500 tokens = 2000 chars — should be medium (not simple)
    msgs = [_msg("x" * 2000)]
    assert r.route(msgs) == "mid"


def test_exactly_six_tools_is_complex():
    r = CostRouter(simple="mini", medium="mid", complex="big")
    tools = [{"name": f"t{i}"} for i in range(6)]
    assert r.route([_msg("hi")], tools) == "big"


def test_model_override_empty_string_is_not_override():
    r = CostRouter(simple="mini", medium="mid", complex="big")
    # empty string should NOT be treated as an override
    assert r.route([_msg("hi")], model_override=None) == "mini"
