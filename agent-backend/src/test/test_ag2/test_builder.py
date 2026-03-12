"""
Unit tests for ag2/builder.py — no LLM calls, no network, no MongoDB.
All external dependencies (mongo, socket, AG2 agents) are mocked.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock
import pytest


# ---------------------------------------------------------------------------
# Helpers to build minimal mock objects that mirror the MongoDB models
# ---------------------------------------------------------------------------

def _make_agent(name="researcher", role="Researcher", goal="Find information.", backstory="Expert."):
    agent = MagicMock()
    agent.name = name
    agent.role = role
    agent.goal = goal
    agent.backstory = backstory
    agent.modelId = "model-id-123"
    agent.toolIds = []
    return agent


def _make_model(model_name="gpt-4o-mini", api_key="test-key", temperature=0.3):
    model = MagicMock()
    model.config = {"model": model_name, "api_key": api_key, "base_url": "https://api.openai.com/v1"}
    model.temperature = temperature
    return model


def _make_crew(ag2_mode="group_chat"):
    crew = MagicMock()
    crew.config = {"ag2_mode": ag2_mode}
    return crew


# ---------------------------------------------------------------------------
# Fixture: patch all external dependencies before AG2Builder is instantiated
# ---------------------------------------------------------------------------

@pytest.fixture()
def builder(request):
    """
    Returns a fully constructed AG2Builder with mocked MongoDB and socket.
    Pass `ag2_mode` via pytest.mark.parametrize or request.param (default: "group_chat").
    """
    mode = getattr(request, "param", "group_chat")
    agent = _make_agent()
    crew = _make_crew(mode)
    model = _make_model()

    mock_mongo = MagicMock()
    mock_mongo.get_session.return_value = MagicMock()
    mock_mongo.get_crew.return_value = (MagicMock(), crew, [], [agent])
    mock_mongo.get_single_model_by_id.return_value = model

    mock_socket = MagicMock()
    mock_socket.connected = True

    with (
        patch("ag2.builder.start_mongo_session", return_value=mock_mongo),
        patch("ag2.builder._mongo", mock_mongo),
        patch("ag2.builder.SimpleClient", return_value=mock_socket),
        patch("ag2.builder.LLMConfig") as mock_llm_config,
    ):
        mock_llm_config.return_value = MagicMock()
        from ag2.builder import AG2Builder
        b = AG2Builder.__new__(AG2Builder)
        b.session_id = "session-test"
        b.socket = mock_socket
        b.agents_config = [agent]
        b.ag2_mode = mode
        b.llm_config = mock_llm_config.return_value
        yield b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAG2BuilderDispatch:
    @pytest.mark.parametrize("mode,method", [
        ("two_agent", "_run_two_agent"),
        ("group_chat", "_run_group_chat"),
        ("reasoning", "_run_reasoning"),
    ])
    def test_run_dispatches_to_correct_handler(self, mode, method):
        agent = _make_agent()
        crew = _make_crew(mode)
        model = _make_model()
        mock_mongo = MagicMock()
        mock_mongo.get_session.return_value = MagicMock()
        mock_mongo.get_crew.return_value = (MagicMock(), crew, [], [agent])
        mock_mongo.get_single_model_by_id.return_value = model
        mock_socket = MagicMock()

        with (
            patch("ag2.builder.start_mongo_session", return_value=mock_mongo),
            patch("ag2.builder._mongo", mock_mongo),
            patch("ag2.builder.SimpleClient", return_value=mock_socket),
            patch("ag2.builder.LLMConfig"),
        ):
            from ag2.builder import AG2Builder
            b = AG2Builder.__new__(AG2Builder)
            b.session_id = "s"
            b.socket = mock_socket
            b.agents_config = [agent]
            b.ag2_mode = mode
            b.llm_config = MagicMock()

            handler = MagicMock()
            with patch.object(b, method, handler):
                with patch.object(b, "_send_stop"):
                    b.run("test message")
            handler.assert_called_once_with("test message")

    def test_unknown_mode_falls_back_to_group_chat(self):
        agent = _make_agent()
        mock_mongo = MagicMock()
        mock_socket = MagicMock()
        crew = _make_crew("nonexistent_mode")
        mock_mongo.get_session.return_value = MagicMock()
        mock_mongo.get_crew.return_value = (MagicMock(), crew, [], [agent])
        mock_mongo.get_single_model_by_id.return_value = _make_model()

        with (
            patch("ag2.builder.start_mongo_session", return_value=mock_mongo),
            patch("ag2.builder._mongo", mock_mongo),
            patch("ag2.builder.SimpleClient", return_value=mock_socket),
            patch("ag2.builder.LLMConfig"),
        ):
            from ag2.builder import AG2Builder
            b = AG2Builder.__new__(AG2Builder)
            b.session_id = "s"
            b.socket = mock_socket
            b.agents_config = [agent]
            b.ag2_mode = "nonexistent_mode"
            b.llm_config = MagicMock()

            group_chat_called = []
            with patch.object(b, "_run_group_chat", side_effect=lambda m: group_chat_called.append(m)):
                with patch.object(b, "_send_stop"):
                    b.run("hello")
            assert group_chat_called == ["hello"]

    def test_send_stop_always_called(self, builder):
        """_send_stop must fire even when the handler raises."""
        with patch.object(builder, "_run_group_chat", side_effect=RuntimeError("boom")):
            with patch.object(builder, "_send_stop") as mock_stop:
                with patch.object(builder, "_send_to_socket"):
                    builder.run("test")
        mock_stop.assert_called_once()


class TestAG2BuilderLLMConfig:
    def test_llm_config_reads_from_model_config(self):
        agent = _make_agent()
        crew = _make_crew("group_chat")
        model = _make_model(model_name="gpt-4o", api_key="sk-secret", temperature=0.5)
        mock_mongo = MagicMock()
        mock_mongo.get_session.return_value = MagicMock()
        mock_mongo.get_crew.return_value = (MagicMock(), crew, [], [agent])
        mock_mongo.get_single_model_by_id.return_value = model
        mock_socket = MagicMock()

        captured_args = {}

        def capture_llm_config(*args, **kwargs):
            captured_args["args"] = args
            captured_args["kwargs"] = kwargs
            return MagicMock()

        with (
            patch("ag2.builder.start_mongo_session", return_value=mock_mongo),
            patch("ag2.builder._mongo", mock_mongo),
            patch("ag2.builder.SimpleClient", return_value=mock_socket),
            patch("ag2.builder.LLMConfig", side_effect=capture_llm_config),
            patch("ag2.builder.AG2Builder._init_socket"),
        ):
            from ag2.builder import AG2Builder
            b = AG2Builder.__new__(AG2Builder)
            b.session_id = "s"
            b.socket = mock_socket
            b._init_app_state()

        config_dict = captured_args["args"][0]
        assert config_dict["model"] == "gpt-4o"
        assert config_dict["api_key"] == "sk-secret"
        assert captured_args["kwargs"]["temperature"] == 0.5


class TestAG2BuilderPlatformTools:
    def test_platform_tools_empty_when_no_tool_ids(self, builder):
        builder.agents_config[0].toolIds = []
        assert builder._platform_tools() == []

    def test_platform_tools_empty_when_no_agents(self, builder):
        builder.agents_config = []
        assert builder._platform_tools() == []


class TestAppTypeEnum:
    def test_ag2_app_type_value(self):
        from models.mongo import AppType
        assert AppType.AG2 == "ag2"
        assert AppType.AG2 in list(AppType)
