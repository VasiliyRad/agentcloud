"""
Pre-mock agentcloud infrastructure modules so ag2.builder can be imported
without a live MongoDB, socket, or full agentcloud stack.
"""
import sys
from enum import Enum
from types import ModuleType
from unittest.mock import MagicMock


class _AppType(str, Enum):
    AG2 = "ag2"
    CREW = "crew"
    LANGGRAPH = "langgraph"


def _stub_module(name: str, **attrs) -> ModuleType:
    m = ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# Stub out agentcloud modules that ag2.builder imports at module level
_stub_module("mongo")
_stub_module("mongo.queries", MongoClientConnection=MagicMock())
_stub_module("init")
_stub_module("init.mongo_session", start_mongo_session=MagicMock())
_stub_module("init.env_variables",
             AGENT_BACKEND_SOCKET_TOKEN="test-token",
             SOCKET_URL="http://localhost:9999")
_stub_module("models")
_stub_module("models.mongo", Agent=MagicMock(), Model=MagicMock(), Session=MagicMock(), AppType=_AppType)
_stub_module("models.sockets",
             Message=MagicMock(),
             SocketEvents=MagicMock(),
             SocketMessage=MagicMock())
_stub_module("socketio", SimpleClient=MagicMock())
_stub_module("socketio.exceptions", ConnectionError=ConnectionError)
_stub_module("messaging")
_stub_module("messaging.send_message_to_socket", send=MagicMock())

# Pre-import local ag2 submodules so patch("ag2.builder.*") can resolve them.
# Python's pkgutil.resolve_name("ag2.builder") does __import__("ag2") then
# getattr(ag2, "builder") — that getattr only works if ag2.builder was already
# imported. The root conftest has already inserted src/ into sys.path.
import ag2.builder  # noqa: E402  (must come after stubs)
import ag2.streaming  # noqa: E402
