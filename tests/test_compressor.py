from tokenshield.compressor import Compressor


def _msgs(n: int) -> list[dict]:
    """Alternating user/assistant messages."""
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"msg {i}"})
    return msgs


def test_history_windowing_keeps_system():
    c = Compressor(max_history_turns=2)
    msgs = [{"role": "system", "content": "sys"}] + _msgs(10)
    result, _ = c.compress(msgs)
    assert result[0]["role"] == "system"
    # 2 turns = 4 messages (2 user + 2 assistant) + system
    assert len(result) == 5


def test_history_windowing_no_op_when_under_limit():
    c = Compressor(max_history_turns=20)
    msgs = _msgs(4)
    result, _ = c.compress(msgs)
    assert result == msgs


def test_tool_pruning():
    c = Compressor(max_tools=2)
    tools = [{"name": f"tool_{i}"} for i in range(5)]
    _, result_tools = c.compress([], tools)
    assert len(result_tools) == 2
    assert result_tools[0]["name"] == "tool_0"


def test_tool_pruning_no_op_when_under_limit():
    c = Compressor(max_tools=10)
    tools = [{"name": "t1"}, {"name": "t2"}]
    _, result_tools = c.compress([], tools)
    assert result_tools == tools


def test_system_prompt_truncation():
    c = Compressor(max_system_tokens=10)  # ~40 chars
    long_system = "x" * 500
    msgs = [{"role": "system", "content": long_system}]
    result, _ = c.compress(msgs)
    assert "[truncated]" in result[0]["content"]
    assert len(result[0]["content"]) < len(long_system)


def test_system_prompt_no_truncation_when_under_limit():
    c = Compressor(max_system_tokens=2000)
    msgs = [{"role": "system", "content": "short"}]
    result, _ = c.compress(msgs)
    assert result[0]["content"] == "short"


def test_none_tools_passthrough():
    c = Compressor()
    _, result_tools = c.compress([], None)
    assert result_tools is None


def test_history_windowing_zero_turns_drops_all_convo():
    c = Compressor(max_history_turns=0)
    msgs = [{"role": "system", "content": "sys"}] + _msgs(4)
    result, _ = c.compress(msgs)
    assert len(result) == 1
    assert result[0]["role"] == "system"
