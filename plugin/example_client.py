"""Minimal host client: speak one line and show emotion.

    python example_client.py "你好呀"
"""

import asyncio
import json
import sys

import websockets

URL = "ws://127.0.0.1:8765"


async def main(text: str) -> None:
    async with websockets.connect(URL) as ws:
        await ws.send(json.dumps({"type": "hello", "client": "example", "protocol": 1}))
        await ws.send(
            json.dumps(
                {
                    "type": "feel",
                    "labels": {"happy": 0.6, "calm": 0.4},
                    "intensity": 0.7,
                    "confidence": 0.8,
                }
            )
        )
        await ws.send(json.dumps({"type": "utterance.start", "utterance_id": "u1"}))
        await ws.send(json.dumps({"type": "text.delta", "text": text}))
        await ws.send(json.dumps({"type": "utterance.end"}))
        await asyncio.sleep(max(2.0, len(text) * 0.18))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "你好呀，我是通过自己的程序接进来的。"))
