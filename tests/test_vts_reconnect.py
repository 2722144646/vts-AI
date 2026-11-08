import asyncio

from dsh_vts_avatar.config import VTSConfig
from dsh_vts_avatar.vts.client import VTSClient


class _DroppingSocket:
    """A socket whose receive loop raises, the way a keepalive timeout looks."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise ConnectionError("keepalive ping timeout")


def test_receive_loop_failure_marks_client_disconnected():
    async def run():
        client = VTSClient(VTSConfig())
        client.ws = _DroppingSocket()
        client.authenticated = True
        await client._receive_loop()
        return client

    client = asyncio.run(run())
    # The engine must not keep believing it can inject after the socket died.
    assert client.connected is False
    assert client.authenticated is False


def test_close_clears_auth_and_connection():
    async def run():
        client = VTSClient(VTSConfig())
        client.authenticated = True
        await client.close()
        return client

    client = asyncio.run(run())
    assert client.connected is False
    assert client.authenticated is False
