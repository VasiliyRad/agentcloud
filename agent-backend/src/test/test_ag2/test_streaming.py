"""Unit tests for ag2/streaming.py — no network, no AG2 LLM calls."""
from unittest.mock import MagicMock, patch
import pytest

# Patch send and models before importing the module under test
with patch("ag2.streaming.send"), patch("ag2.streaming.SocketEvents"), patch("ag2.streaming.SocketMessage"), patch("ag2.streaming.Message"):
    from ag2.streaming import make_socket_reply_func


@pytest.fixture()
def socket():
    return MagicMock()


@pytest.fixture()
def recipient():
    m = MagicMock()
    m.name = "assistant"
    return m


class TestMakeSocketReplyFunc:
    def test_returns_callable(self, socket):
        with patch("ag2.streaming.send"), patch("ag2.streaming.SocketEvents"), \
             patch("ag2.streaming.SocketMessage"), patch("ag2.streaming.Message"):
            fn = make_socket_reply_func(socket, "s1")
        assert callable(fn)

    def test_reply_func_signature_returns_false_none(self, socket, recipient):
        """reply_func must return (False, None) so AG2 continues its normal flow."""
        with (
            patch("ag2.streaming.send"),
            patch("ag2.streaming.SocketEvents"),
            patch("ag2.streaming.SocketMessage"),
            patch("ag2.streaming.Message"),
        ):
            fn = make_socket_reply_func(socket, "s1")
            result = fn(recipient, [{"content": "Hello!"}], None, None)
        assert result == (False, None)

    def test_emits_on_non_empty_content(self, socket, recipient):
        with (
            patch("ag2.streaming.send") as mock_send,
            patch("ag2.streaming.SocketEvents"),
            patch("ag2.streaming.SocketMessage"),
            patch("ag2.streaming.Message"),
        ):
            fn = make_socket_reply_func(socket, "session-abc")
            fn(recipient, [{"content": "Here is the answer."}], None, None)

        mock_send.assert_called_once()

    def test_silent_on_empty_content(self, socket, recipient):
        with (
            patch("ag2.streaming.send") as mock_send,
            patch("ag2.streaming.SocketEvents"),
            patch("ag2.streaming.SocketMessage"),
            patch("ag2.streaming.Message"),
        ):
            fn = make_socket_reply_func(socket, "session-abc")
            fn(recipient, [{"content": ""}], None, None)

        mock_send.assert_not_called()

    def test_silent_on_empty_messages(self, socket, recipient):
        with (
            patch("ag2.streaming.send") as mock_send,
            patch("ag2.streaming.SocketEvents"),
            patch("ag2.streaming.SocketMessage"),
            patch("ag2.streaming.Message"),
        ):
            fn = make_socket_reply_func(socket, "session-abc")
            fn(recipient, [], None, None)

        mock_send.assert_not_called()

    def test_uses_recipient_name_as_author(self, socket):
        recipient = MagicMock()
        recipient.name = "researcher"
        captured = {}

        def capture_send(sock, event, msg, mode):
            captured["msg"] = msg

        with (
            patch("ag2.streaming.send", side_effect=capture_send),
            patch("ag2.streaming.SocketEvents"),
            patch("ag2.streaming.SocketMessage", side_effect=lambda **kw: kw) as mock_sm,
            patch("ag2.streaming.Message"),
        ):
            fn = make_socket_reply_func(socket, "session-abc")
            fn(recipient, [{"content": "Found it."}], None, None)

        assert mock_sm.call_args.kwargs["authorName"] == "researcher"
