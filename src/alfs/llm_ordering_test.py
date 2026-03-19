from alfs.llm_ordering import can_overwrite, rank


def test_rank_known_models():
    assert rank("qwen2.5:32b") == 1
    assert rank("claude-code") == 2


def test_rank_unknown_model():
    assert rank("some-unknown-model") == 0


def test_rank_none():
    assert rank(None) == 0


def test_rank_ordering():
    assert rank("qwen2.5:32b") < rank("claude-code")


def test_can_overwrite_higher_rank():
    assert can_overwrite("claude-code", "qwen2.5:32b") is True


def test_can_overwrite_same_rank():
    assert can_overwrite("claude-code", "claude-code") is True
    assert can_overwrite("qwen2.5:32b", "qwen2.5:32b") is True


def test_can_overwrite_lower_rank():
    assert can_overwrite("qwen2.5:32b", "claude-code") is False


def test_can_overwrite_unknown_requesting():
    assert can_overwrite("unknown-model", "qwen2.5:32b") is False
    assert can_overwrite("unknown-model", "claude-code") is False


def test_can_overwrite_none_requesting():
    assert can_overwrite(None, "qwen2.5:32b") is False
    assert can_overwrite(None, "claude-code") is False


def test_can_overwrite_none_existing():
    assert can_overwrite(None, None) is True
    assert can_overwrite("qwen2.5:32b", None) is True
    assert can_overwrite("claude-code", None) is True


def test_can_overwrite_unknown_existing():
    assert can_overwrite(None, "unknown-model") is True
    assert can_overwrite("qwen2.5:32b", "unknown-model") is True
