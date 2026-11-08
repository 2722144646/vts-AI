"""Local WebSocket bridge consumed by the dsh Host plugin."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import orjson
import websockets
from websockets.asyncio.server import Server, ServerConnection, serve

from ..config import Config
from .engine import AvatarEngine

log = logging.getLogger(__name__)


class BridgeServer:
    def __init__(self, config: Config):
        self.config = config
        self.engine = AvatarEngine(config, on_event=self.broadcast)
        self.clients: set[ServerConnection] = set()
        self._server: Server | None = None
        self._send_lock = asyncio.Lock()

    async def start(self) -> None:
        await self.engine.start()
        self._server = await serve(
            self._handler,
            self.config.bridge.host,
            self.config.bridge.port,
            max_size=2 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        )
        log.info("bridge listening on ws://%s:%d", self.config.bridge.host, self.config.bridge.port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        await self.engine.close()

    async def broadcast(self, payload: dict[str, Any]) -> None:
        if not self.clients:
            return
        raw = orjson.dumps(payload).decode("utf-8")
        async with self._send_lock:
            stale: list[ServerConnection] = []
            for client in list(self.clients):
                try:
                    await client.send(raw)
                except Exception:
                    stale.append(client)
            for client in stale:
                self.clients.discard(client)

    async def _handler(self, ws: ServerConnection) -> None:
        self.clients.add(ws)
        peer = getattr(ws, "remote_address", None)
        log.info("plugin connected: %s", peer)
        try:
            await ws.send(
                orjson.dumps(
                    {
                        "type": "hello.ack",
                        "protocol": 1,
                        "status": self.engine.status(),
                    }
                ).decode("utf-8")
            )
            async for raw in ws:
                try:
                    msg = orjson.loads(raw)
                except Exception:
                    await ws.send(orjson.dumps({"type": "error", "code": "bad_json"}).decode("utf-8"))
                    continue
                try:
                    await self._dispatch(ws, msg)
                except Exception as exc:
                    log.exception("bridge dispatch failed")
                    await ws.send(
                        orjson.dumps(
                            {
                                "type": "error",
                                "code": "dispatch_failed",
                                "message": str(exc),
                                "retryable": True,
                            }
                        ).decode("utf-8")
                    )
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)
            log.info("plugin disconnected: %s", peer)

    async def _dispatch(self, ws: ServerConnection, msg: dict[str, Any]) -> None:
        mtype = str(msg.get("type", ""))
        if mtype == "hello":
            await ws.send(
                orjson.dumps(
                    {
                        "type": "ready",
                        "protocol": 1,
                        "status": self.engine.status(),
                    }
                ).decode("utf-8")
            )
        elif mtype == "utterance.start":
            log.info("utterance.start %s", msg.get("utterance_id"))
            await self.engine.begin_utterance(str(msg.get("utterance_id") or ""))
            await ws.send(
                orjson.dumps(
                    {"type": "utterance.started", "utterance_id": self.engine.utterance_id}
                ).decode("utf-8")
            )
        elif mtype == "text.delta":
            text = str(msg.get("text") or "")
            log.info("text.delta %d chars", len(text))
            await self.engine.feed_text(text)
        elif mtype == "utterance.end":
            log.info("utterance.end %s", msg.get("utterance_id"))
            await self.engine.end_utterance()
        elif mtype == "cancel":
            await self.engine.cancel()
        elif mtype == "state":
            self.engine.set_state(str(msg.get("state") or "idle"))
        elif mtype == "action":
            self.engine.add_action(str(msg.get("kind") or "nod"), float(msg.get("intensity", 0.7)))
        elif mtype == "emotion":
            self.engine.set_emotion(
                str(msg.get("name") or "neutral"),
                float(msg.get("intensity", 0.7)),
            )
        elif mtype == "feel":
            # Phase 3: weighted labels + confidence + note. Phase 1 of emotion
            # memory also carries cause/why, captured at the moment of feeling.
            result = self.engine.feel(
                labels=msg.get("labels") if isinstance(msg.get("labels"), dict) else {},
                intensity=float(msg.get("intensity", 0.7)),
                confidence=float(msg.get("confidence", 0.8)),
                note=str(msg.get("note") or ""),
                turn_id=str(msg.get("turn_id") or ""),
                cause=str(msg.get("cause") or ""),
                why=str(msg.get("why") or ""),
                appraisal=msg.get("appraisal") if isinstance(msg.get("appraisal"), list) else None,
                display=float(msg["display"]) if msg.get("display") is not None else None,
            )
            await ws.send(
                orjson.dumps(
                    {"type": "feel.result", "request_id": msg.get("request_id"), **result}
                ).decode("utf-8")
            )
        elif mtype == "mood.get":
            # Emotion memory: current mood plus the reasons behind it.
            await ws.send(
                orjson.dumps(
                    {
                        "type": "mood.result",
                        "request_id": msg.get("request_id"),
                        **self.engine.mood_payload(),
                    }
                ).decode("utf-8")
            )
        elif mtype == "llm.get":
            await ws.send(
                orjson.dumps(
                    {"type": "llm.get.result", "request_id": msg.get("request_id"), **self.engine.llm_settings()}
                ).decode("utf-8")
            )
        elif mtype == "llm.set":
            result = self.engine.save_llm_settings(
                msg.get("values") if isinstance(msg.get("values"), dict) else {}
            )
            await ws.send(
                orjson.dumps(
                    {"type": "llm.set.result", "request_id": msg.get("request_id"), **result}
                ).decode("utf-8")
            )
        elif mtype == "settings.get":
            await ws.send(orjson.dumps({"type": "settings.get.result", "request_id": msg.get("request_id"), **self.engine.settings_snapshot()}).decode("utf-8"))
        elif mtype == "settings.set":
            result = self.engine.save_settings(msg.get("values") if isinstance(msg.get("values"), dict) else {})
            await ws.send(orjson.dumps({"type": "settings.set.result", "request_id": msg.get("request_id"), **result}).decode("utf-8"))
        elif mtype == "diary.folder":
            await ws.send(orjson.dumps({"type": "diary.folder.result", "request_id": msg.get("request_id"), **self.engine.open_diary_folder()}).decode("utf-8"))
        elif mtype == "idle.request":
            await ws.send(orjson.dumps({"type": "idle.request.result", "request_id": msg.get("request_id"), **self.engine.request_idle_write()}).decode("utf-8"))
        elif mtype == "audio.calibrate":
            result = await self.engine.recalibrate_audio()
            await ws.send(orjson.dumps({"type": "audio.calibrate.result", "request_id": msg.get("request_id"), **result}).decode("utf-8"))
        elif mtype == "panel.open":
            result = self.engine.open_panel()
            await ws.send(
                orjson.dumps(
                    {"type": "panel.open.result", "request_id": msg.get("request_id"), **result}
                ).decode("utf-8")
            )
        elif mtype == "transcript.window":
            result = self.engine.open_transcript_window()
            await ws.send(
                orjson.dumps(
                    {"type": "transcript.window.result", "request_id": msg.get("request_id"), **result}
                ).decode("utf-8")
            )
        elif mtype == "local.chat":
            result = await self.engine.local_reply(
                str(msg.get("text") or ""), show_text=bool(msg.get("show_text"))
            )
            await ws.send(
                orjson.dumps(
                    {"type": "local.chat.result", "request_id": msg.get("request_id"), **result}
                ).decode("utf-8")
            )
        elif mtype == "idle.write":
            actions = msg.get("actions") or []
            if not isinstance(actions, list):
                actions = []
            result = self.engine.idle_write(
                str(msg.get("profile") or ""), actions, str(msg.get("note") or "")
            )
            await ws.send(
                orjson.dumps(
                    {"type": "idle.write.result", "request_id": msg.get("request_id"), **result}
                ).decode("utf-8")
            )
        elif mtype == "user.switch":
            result = self.engine.switch_user(
                str(msg.get("user_id") or ""), str(msg.get("display_name") or "")
            )
            await ws.send(
                orjson.dumps(
                    {"type": "user.result", "request_id": msg.get("request_id"), **result}
                ).decode("utf-8")
            )
        elif mtype == "relationship.event":
            result = self.engine.relationship_event(
                str(msg.get("event") or ""), str(msg.get("note") or "")
            )
            await ws.send(
                orjson.dumps(
                    {"type": "relationship.result", "request_id": msg.get("request_id"), **result}
                ).decode("utf-8")
            )
        elif mtype == "diary.write":
            result = self.engine.diary_write(str(msg.get("text") or ""))
            await ws.send(
                orjson.dumps(
                    {"type": "diary.write.result", "request_id": msg.get("request_id"), **result}
                ).decode("utf-8")
            )
        elif mtype == "private":
            result = self.engine.private_action(
                action=str(msg.get("action") or "list"),
                content=str(msg.get("content") or ""),
                kind=str(msg.get("kind") or "feeling"),
                privacy=float(msg.get("privacy", 0.5)),
                item_id=str(msg.get("item_id") or ""),
            )
            await ws.send(
                orjson.dumps(
                    {"type": "private.result", "request_id": msg.get("request_id"), **result}
                ).decode("utf-8")
            )
        elif mtype == "recall":
            result = self.engine.recall(
                query=str(msg.get("query") or ""),
                count=int(msg.get("count", 3)),
            )
            await ws.send(
                orjson.dumps(
                    {"type": "recall.result", "request_id": msg.get("request_id"), **result}
                ).decode("utf-8")
            )
        elif mtype == "explain":            # Phase 2: how an explanation landed (model judges acceptance).
            result = self.engine.explain(
                explanation=str(msg.get("explanation") or ""),
                acceptance=float(msg["acceptance"]) if msg.get("acceptance") is not None else None,
                status=str(msg.get("status") or ""),
                defense=str(msg.get("defense") or "") or None,
                episode_id=str(msg.get("episode_id") or ""),
            )
            await ws.send(
                orjson.dumps(
                    {"type": "explain.result", "request_id": msg.get("request_id"), **result}
                ).decode("utf-8")
            )
        elif mtype == "record":
            summary = await self.engine.record_expression(
                str(msg.get("name") or "learned"),
                float(msg.get("seconds", 4.0)),
            )
            await ws.send(
                orjson.dumps(
                    {
                        "type": "record.result",
                        "request_id": msg.get("request_id"),
                        "summary": summary,
                    }
                ).decode("utf-8")
            )
        elif mtype == "head_demo":
            ok = self.engine.demo_head(
                curve=str(msg.get("curve") or "minjerk"),
                intensity=float(msg.get("intensity", 0.5)),
                duration_ms=int(msg.get("duration_ms", 900)),
            )
            await ws.send(
                orjson.dumps({"type": "head_demo.result", "ok": ok}).decode("utf-8")
            )
        elif mtype == "reconnect":
            ok = await self.engine.reconnect()
            await ws.send(orjson.dumps({"type": "reconnected", "ok": ok}).decode("utf-8"))
        elif mtype == "snapshot":
            metrics = await self.engine.snapshot(
                frames=max(1, min(12, int(msg.get("frames", 1)))),
                interval_ms=int(msg.get("interval_ms", 180)),
            )
            await ws.send(
                orjson.dumps(
                    {
                        "type": "snapshot.result",
                        "request_id": msg.get("request_id"),
                        "metrics": metrics,
                    }
                ).decode("utf-8")
            )
        elif mtype == "head_pose":
            ok = self.engine.set_head_pose(
                yaw=float(msg.get("yaw", 0.0)),
                pitch=float(msg.get("pitch", 0.0)),
                roll=float(msg.get("roll", 0.0)),
                seconds=float(msg.get("seconds", 8.0)),
            )
            await ws.send(orjson.dumps({"type": "head_pose.result", "ok": ok}).decode("utf-8"))
        elif mtype == "parameters":
            await ws.send(
                orjson.dumps(
                    {
                        "type": "parameters.result",
                        "request_id": msg.get("request_id"),
                        "data": self.engine.parameters(),
                    }
                ).decode("utf-8")
            )
        elif mtype == "status":
            await ws.send(orjson.dumps({"type": "status", "status": self.engine.status()}).decode("utf-8"))
        elif mtype == "ping":
            await ws.send(orjson.dumps({"type": "pong"}).decode("utf-8"))
        else:
            await ws.send(
                orjson.dumps({"type": "error", "code": "unknown_type", "message": mtype}).decode("utf-8")
            )


async def run_bridge(config: Config) -> None:
    server = BridgeServer(config)
    await server.start()
    try:
        await asyncio.Future()
    finally:
        await server.stop()
