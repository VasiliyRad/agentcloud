"""
AG2 (formerly AutoGen) runtime for AgentCloud.

Supports three orchestration modes (set via crew.config["ag2_mode"]):
  two_agent   — AssistantAgent + UserProxyAgent (Docker sandbox for code execution)
  group_chat  — GroupChat with LLM-driven speaker selection
  reasoning   — ReasoningAgent with tree-of-thought beam search

Usage: AG2Builder("session-id").run("user message")
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime

from autogen import (
    AssistantAgent,
    ConversableAgent,
    GroupChat,
    GroupChatManager,
    LLMConfig,
    UserProxyAgent,
)
from autogen.agents.experimental import ReasoningAgent
from socketio import SimpleClient
from socketio.exceptions import ConnectionError as ConnError

from ag2.streaming import make_socket_reply_func
from ag2.tool_adapter import build_ag2_tools
from init.env_variables import AGENT_BACKEND_SOCKET_TOKEN, SOCKET_URL
from init.mongo_session import start_mongo_session
from models.mongo import Agent, Model, Session
from models.sockets import Message, SocketEvents, SocketMessage
from messaging.send_message_to_socket import send

logger = logging.getLogger(__name__)

_mongo = start_mongo_session()


class AG2Builder:
    """
    Builds and runs an AG2 multi-agent session.
    Reads agent/model config from MongoDB; follows the ChatAssistant pattern.
    ag2_mode is stored in crew.config["ag2_mode"] (defaults to "group_chat").
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.socket = SimpleClient()
        self._init_socket()
        self._init_app_state()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_socket(self) -> None:
        try:
            custom_headers = {"x-agent-backend-socket-token": AGENT_BACKEND_SOCKET_TOKEN}
            self.socket.connect(url=SOCKET_URL, headers=custom_headers)
            self.socket.emit("join_room", f"_{self.session_id}")
        except ConnError as ce:
            logger.error("Socket connection error: %s", ce)
            raise

    def _init_app_state(self) -> None:
        session: Session = _mongo.get_session(self.session_id)
        _app, the_crew, _tasks, crew_agents = _mongo.get_crew(session)

        self.agents_config: list[Agent] = list(crew_agents) if crew_agents else []
        self.ag2_mode: str = (the_crew.config or {}).get("ag2_mode", "group_chat")

        # Build shared LLMConfig from the first agent's model
        primary = self.agents_config[0] if self.agents_config else None
        if primary:
            model: Model = _mongo.get_single_model_by_id("models", Model, primary.modelId)
            cfg = model.config or {}
            self.llm_config = LLMConfig(
                {
                    "model": cfg.get("model", "gpt-4o-mini"),
                    "api_key": cfg.get("api_key", ""),
                    "base_url": cfg.get("base_url", "https://api.openai.com/v1"),
                },
                temperature=model.temperature if model.temperature is not None else 0.3,
                cache_seed=None,
            )
        else:
            self.llm_config = None

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def run(self, message: str) -> None:
        handler = {
            "two_agent": self._run_two_agent,
            "group_chat": self._run_group_chat,
            "reasoning": self._run_reasoning,
        }.get(self.ag2_mode, self._run_group_chat)
        try:
            handler(message)
        except Exception as exc:
            logger.exception("AG2 runtime error in session %s", self.session_id)
            self._send_to_socket(f"AG2 runtime error: {exc}")
        finally:
            self._send_stop()

    # ── Two-agent mode ────────────────────────────────────────────────────────

    def _run_two_agent(self, message: str) -> None:
        reply_fn = make_socket_reply_func(self.socket, self.session_id)
        system_msg = (
            "\n".join([self.agents_config[0].role, self.agents_config[0].goal])
            if self.agents_config
            else "You are a helpful AI assistant."
        )
        tools = build_ag2_tools(self._platform_tools())

        assistant = AssistantAgent(
            name="assistant",
            system_message=system_msg,
            llm_config=self.llm_config,
        )
        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            # use_docker=True: agent-generated code runs in an isolated container,
            # not in the platform's Python process. Required for a managed platform.
            # Ensure the Docker socket is accessible from the agent-backend container.
            code_execution_config={"work_dir": f"/tmp/ag2_{self.session_id}", "use_docker": True},
            is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
        )
        for tool_fn, tool_meta in tools:
            assistant.register_for_llm(**tool_meta)(tool_fn)
            user_proxy.register_for_execution(**tool_meta)(tool_fn)

        assistant.register_reply(trigger=ConversableAgent, reply_func=reply_fn, position=0)
        user_proxy.initiate_chat(assistant, message=message)

    # ── Group chat mode ───────────────────────────────────────────────────────

    def _run_group_chat(self, message: str) -> None:
        reply_fn = make_socket_reply_func(self.socket, self.session_id)
        tools = build_ag2_tools(self._platform_tools())

        agents = [
            AssistantAgent(
                name=a.name.replace(" ", "_"),
                system_message="\n".join([a.role, a.goal, a.backstory]),
                llm_config=self.llm_config,
            )
            for a in self.agents_config
        ] or [
            AssistantAgent(
                name="assistant",
                system_message="You are a helpful assistant.",
                llm_config=self.llm_config,
            )
        ]

        executor = UserProxyAgent(
            name="executor",
            human_input_mode="NEVER",
            code_execution_config=False,
            is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
        )
        for tool_fn, tool_meta in tools:
            agents[0].register_for_llm(**tool_meta)(tool_fn)
            executor.register_for_execution(**tool_meta)(tool_fn)

        for agent in agents:
            agent.register_reply(trigger=ConversableAgent, reply_func=reply_fn, position=0)

        gc = GroupChat(
            agents=[executor] + agents,
            messages=[],
            max_round=15,
            speaker_selection_method="auto",
        )
        manager = GroupChatManager(groupchat=gc, llm_config=self.llm_config)
        executor.initiate_chat(manager, message=message)

    # ── Reasoning mode ────────────────────────────────────────────────────────

    def _run_reasoning(self, message: str) -> None:
        reply_fn = make_socket_reply_func(self.socket, self.session_id)
        system_msg = (
            "\n".join([self.agents_config[0].role, self.agents_config[0].goal])
            if self.agents_config
            else "You are a careful reasoning agent."
        )
        agent = ReasoningAgent(
            name="reasoning_agent",
            system_message=system_msg,
            llm_config=self.llm_config,
            reason_config={"method": "beam_search", "beam_size": 3, "max_depth": 4},
        )
        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            code_execution_config=False,
            is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
        )
        agent.register_reply(trigger=ConversableAgent, reply_func=reply_fn, position=0)
        user_proxy.initiate_chat(agent, message=message)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _platform_tools(self) -> list[dict]:
        """
        Returns tool configs as {name, description, fn} dicts for all tools
        attached to the first agent in the crew. Extend here to support
        per-agent tool assignment in group_chat mode.
        """
        if not self.agents_config or not self.agents_config[0].toolIds:
            return []
        from models.mongo import Tool
        tools = _mongo.get_models_by_ids("tools", Tool, self.agents_config[0].toolIds) or []
        result = []
        for tool in tools:
            # Wrap tool.run as a plain callable for AG2 registration
            def _make_fn(t):
                def fn(query: str) -> str:
                    return t.run(query)
                fn.__name__ = t.name.replace(" ", "_").lower()
                fn.__doc__ = t.description or t.name
                return fn
            result.append({
                "name": tool.name.replace(" ", "_").lower(),
                "description": tool.description or tool.name,
                "fn": _make_fn(tool),
            })
        return result

    def _send_to_socket(self, text: str) -> None:
        send(
            self.socket,
            SocketEvents.MESSAGE,
            SocketMessage(
                room=self.session_id,
                authorName="AG2",
                message=Message(
                    chunkId=str(uuid.uuid4()),
                    text=text,
                    first=True,
                    tokens=1,
                    timestamp=int(datetime.now().timestamp() * 1000),
                    displayType="bubble",
                ),
            ),
            "both",
        )

    def _send_stop(self) -> None:
        send(
            self.socket,
            SocketEvents.STOP_GENERATING,
            SocketMessage(
                room=self.session_id,
                authorName="AG2",
                message=Message(
                    chunkId=str(uuid.uuid4()),
                    text="",
                    first=True,
                    tokens=0,
                    timestamp=int(datetime.now().timestamp() * 1000),
                    displayType="bubble",
                ),
            ),
            "both",
        )
