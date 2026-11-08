import asyncio
import json

import websockets
from websockets.asyncio.server import serve

from dsh_vts_avatar.vts.client import find_vts_port


def _responder():
    async def handler(ws):
        async for raw in ws:
            message = json.loads(raw)
            if message.get("messageType") == "APIStateRequest":
                await ws.send(
                    json.dumps(
                        {
                            "apiName": "VTubeStudioPublicAPI",
                            "messageType": "APIStateResponse",
                            "requestID": message["requestID"],
                            "data": {"apiVersion": "1.0"},
                        }
                    )
                )

    return handler


def test_find_vts_port_accepts_only_api_state_responses():
    async def run():
        async with serve(_responder(), "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            found = await find_vts_port("127.0.0.1", ports=range(port, port + 1))
            skipped = await find_vts_port(
                "127.0.0.1", exclude=port, ports=range(port, port + 1)
            )
            refused = await find_vts_port("127.0.0.1", ports=range(port + 1, port + 2))
        return found, skipped, refused

    found, skipped, refused = asyncio.run(run())
    assert found is not None
    assert skipped is None  # the configured port is not probed again
    assert refused is None  # nothing listening on the next port


def test_find_vts_port_returns_none_when_nothing_answers():
    async def run():
        return await find_vts_port("127.0.0.1", ports=range(1))

    assert asyncio.run(run()) is None
