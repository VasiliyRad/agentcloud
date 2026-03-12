"""
Socket.IO streaming hook for AG2 agents.
Injects a reply_func that forwards complete agent messages to the webapp.

AG2 also supports token-level streaming via ConversableAgent.generate_reply hooks,
which would allow the webapp to render partial tokens as they arrive. That can be
enabled here once the platform's Socket.IO protocol is extended to support
incremental message chunks.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from socketio import SimpleClient

from messaging.send_message_to_socket import send
from models.sockets import Message, SocketEvents, SocketMessage


def make_socket_reply_func(socket: SimpleClient, session_id: str):
    """
    Returns an AG2 reply_func that emits each complete agent message over Socket.IO.
    Register as: agent.register_reply(trigger=ConversableAgent, reply_func=fn, position=0)
    """
    def reply_func(recipient: Any, messages: list, sender: Any, config: Any):
        last = messages[-1] if messages else {}
        content = last.get("content", "")
        if content:
            send(
                socket,
                SocketEvents.MESSAGE,
                SocketMessage(
                    room=session_id,
                    authorName=recipient.name,
                    message=Message(
                        chunkId=str(uuid.uuid4()),
                        text=str(content),
                        first=True,
                        tokens=1,
                        timestamp=int(datetime.now().timestamp() * 1000),
                        displayType="bubble",
                    ),
                ),
                "both",
            )
        return False, None  # don't intercept — let AG2 continue normal reply flow

    return reply_func
