"""Unit tests for ag2/tool_adapter.py — no imports beyond stdlib."""
from ag2.tool_adapter import build_ag2_tools


class TestBuildAg2Tools:
    def test_empty_input_returns_empty_list(self):
        assert build_ag2_tools([]) == []

    def test_returns_list_of_tuples(self):
        fn = lambda q: "result"
        configs = [{"name": "search", "description": "Search the web.", "fn": fn}]
        result = build_ag2_tools(configs)
        assert len(result) == 1
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2

    def test_callable_is_preserved(self):
        fn = lambda q: "result"
        configs = [{"name": "search", "description": "Search the web.", "fn": fn}]
        result_fn, _ = build_ag2_tools(configs)[0]
        assert result_fn is fn

    def test_metadata_keys_are_correct(self):
        fn = lambda q: "result"
        configs = [{"name": "search", "description": "Search the web.", "fn": fn}]
        _, meta = build_ag2_tools(configs)[0]
        assert meta["name"] == "search"
        assert meta["description"] == "Search the web."

    def test_multiple_tools_all_returned(self):
        fn1 = lambda q: "r1"
        fn2 = lambda q: "r2"
        configs = [
            {"name": "tool_a", "description": "A tool.", "fn": fn1},
            {"name": "tool_b", "description": "B tool.", "fn": fn2},
        ]
        result = build_ag2_tools(configs)
        assert len(result) == 2
        assert result[0][0] is fn1
        assert result[1][0] is fn2
