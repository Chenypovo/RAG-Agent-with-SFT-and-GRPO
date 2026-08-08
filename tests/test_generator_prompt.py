from app.generator.generator import compose_user_prompt

MEMORY_HEADER = "About the user"


def test_includes_query_and_context():
    up = compose_user_prompt("What is FAISS?", "[Chunk 1] FAISS is a library", "")
    assert "What is FAISS?" in up
    assert "FAISS is a library" in up


def test_memory_block_injected_when_present():
    up = compose_user_prompt("hi", "ctx", "- likes green tea\n- lives in Singapore")
    assert MEMORY_HEADER in up
    assert "likes green tea" in up


def test_no_memory_section_when_empty_backward_compatible():
    up = compose_user_prompt("hi", "ctx", "")
    assert MEMORY_HEADER not in up


def test_tool_outputs_are_injected_separately_from_memory():
    up = compose_user_prompt("total?", "", "", "[Tool calculator]\n20 + 20 = 40")
    assert "Verified tool outputs" in up
    assert "20 + 20 = 40" in up
    assert MEMORY_HEADER not in up


def test_confirmed_actions_are_separate_and_not_described_as_evidence():
    up = compose_user_prompt(
        "记住我的偏好",
        "",
        "",
        "",
        "[Tool write_memory]\nmemory merged: 1 add",
    )
    assert "Confirmed actions" in up
    assert "operation status only, not factual evidence" in up
    assert "memory merged: 1 add" in up
    assert "Verified tool outputs" not in up
