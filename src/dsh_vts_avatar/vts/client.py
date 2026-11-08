"""Small, dependency-light VTube Studio public API client.

The API injects *tracking parameters* such as ``FaceAngleX`` and ``MouthOpen``.
It does not inject Live2D parameter IDs such as ``ParamAngleX``; those are the
model-side targets that VTS maps to.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import time
import uuid
from typing import Any

import orjson
import websockets
from websockets.asyncio.client import ClientConnection

from ..config import VTSConfig

log = logging.getLogger(__name__)

API_NAME = "VTubeStudioPublicAPI"
API_VERSION = "1.0"

# Errors worth handling specially.
ERR_AUTH_REQUIRED = 8
ERR_TOKEN_DENIED = 50
ERR_TOKEN_REQUEST_ONGOING = 51
ERR_PARAM_NOT_FOUND = 453
ERR_PARAM_CONTROLLED = 454
ERR_MODE_UNKNOWN = 455


class VTSError(RuntimeError):
    """A VTS APIError response."""

    def __init__(self, error_id: int, message: str, message_type: str = "APIError"):
        super().__init__(f"VTS error {error_id}: {message}")
        self.error_id = error_id
        self.vts_message = message
        self.message_type = message_type


class VTSMessageError(RuntimeError):
    """Malformed or unexpected VTS response."""


async def find_vts_port(
    host: str,
    exclude: int | None = None,
    ports: range | None = None,
) -> int | None:
    """Locate VTS after a restart moved its API port.

    VTS reuses the last free port in its configured range; a bridge hard-wired
    to 8002 goes deaf the moment VTS comes back on 8003. Probe the usual band
    and only accept a socket that answers ``APIStateRequest``.
    """
    candidates = [port for port in (ports or range(8001, 8011)) if port != exclude]
    payload = orjson.dumps(
        {
            "apiName": API_NAME,
            "apiVersion": API_VERSION,
            "messageType": "APIStateRequest",
            "requestID": _new_request_id(),
        }
    ).decode("utf-8")
    for port in candidates:
        url = f"ws://{host}:{port}"
        try:
            async with websockets.connect(
                url,
                open_timeout=0.5,
                close_timeout=0.2,
                max_size=1024 * 1024,
            ) as ws:
                await ws.send(payload)
                raw = await asyncio.wait_for(ws.recv(), timeout=0.8)
                message = orjson.loads(raw)
                if message.get("messageType") == "APIStateResponse":
                    log.info("VTS found on port %d", port)
                    return port
        except Exception:
            continue
    return None


def _new_request_id() -> str:
    # VTS allows 1..64 ASCII chars.
    return f"dsh-{uuid.uuid4().hex[:24]}"


class VTSClient:
    def __init__(self, config: VTSConfig, *, logger: logging.Logger | None = None):
        self.config = config
        self.log = logger or log
        self.ws: ClientConnection | None = None
        self._receiver: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._closed = False
        self._inject_errors: list[VTSError] = []
        self.last_response_ms: float | None = None
        self.authenticated = False

    # -- lifecycle ---------------------------------------------------------

    async def connect(self, *, request_token: bool = False) -> None:
        self.log.info("connecting to %s", self.config.url)
        self.ws = await websockets.connect(
            self.config.url,
            max_size=8 * 1024 * 1024,
            open_timeout=5,
            close_timeout=2,
            ping_interval=20,
            ping_timeout=20,
        )
        self._closed = False
        self._receiver = asyncio.create_task(self._receive_loop(), name="vts-receiver")
        state = await self.request("APIStateRequest")
        data = state.get("data") or {}
        if data.get("currentSessionAuthenticated"):
            self.authenticated = True
            self.log.info("VTS session already authenticated")
            return
        await self._authenticate(request_token=request_token)

    async def close(self) -> None:
        self._closed = True
        self.authenticated = False
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
        if self._receiver is not None:
            self._receiver.cancel()
            try:
                await self._receiver
            except (asyncio.CancelledError, Exception):
                pass
        self.ws = None
        self._receiver = None
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()

    @property
    def connected(self) -> bool:
        return self.ws is not None and not self._closed

    # -- request/response --------------------------------------------------

    async def _receive_loop(self) -> None:
        assert self.ws is not None
        try:
            async for raw in self.ws:
                try:
                    msg = orjson.loads(raw)
                except Exception:
                    self.log.warning("VTS sent non-JSON message: %r", raw[:200])
                    continue
                self._handle_message(msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closed:
                self.log.warning("VTS receive loop stopped: %s", exc)
                # The socket is gone even though close() was never called.
                # Without this the engine would keep believing it can inject.
                self._closed = True
                self.authenticated = False
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(VTSMessageError("VTS connection closed"))
            self._pending.clear()

    def _handle_message(self, msg: dict[str, Any]) -> None:
        rid = msg.get("requestID")
        mtype = msg.get("messageType", "")
        if mtype == "APIError":
            data = msg.get("data") or {}
            err = VTSError(int(data.get("errorID", 0)), str(data.get("message", "")), mtype)
            if rid and rid in self._pending:
                fut = self._pending.pop(rid)
                if not fut.done():
                    fut.set_exception(err)
            else:
                self._inject_errors.append(err)
                self.log.warning("VTS APIError: %s", err)
            return
        if rid and rid in self._pending:
            fut = self._pending.pop(rid)
            if not fut.done():
                ts = msg.get("timestamp")
                if isinstance(ts, (int, float)):
                    self.last_response_ms = float(ts)
                fut.set_result(msg)
        else:
            # Event messages or fire-and-forget responses.
            if mtype.endswith("Response"):
                self.log.debug("VTS unsolicited response %s", mtype)

    async def _send(self, payload: dict[str, Any]) -> None:
        if self.ws is None or self._closed:
            raise VTSMessageError("VTS is not connected")
        await self.ws.send(orjson.dumps(payload).decode("utf-8"))

    async def request(
        self,
        message_type: str,
        data: dict[str, Any] | None = None,
        *,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        rid = _new_request_id()
        payload: dict[str, Any] = {
            "apiName": API_NAME,
            "apiVersion": API_VERSION,
            "messageType": message_type,
            "requestID": rid,
        }
        if data is not None:
            payload["data"] = data
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        try:
            await self._send(payload)
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(rid, None)

    # -- auth --------------------------------------------------------------

    async def _authenticate(self, *, request_token: bool) -> None:
        token = self._load_token()
        if token is None or request_token:
            token = await self._request_token()
        resp = await self.request(
            "AuthenticationRequest",
            {
                "pluginName": self.config.plugin_name,
                "pluginDeveloper": self.config.plugin_developer,
                "authenticationToken": token,
            },
        )
        data = resp.get("data") or {}
        if not data.get("authenticated"):
            # Token may have been revoked; ask for a fresh one once.
            if not request_token:
                self.log.warning("stored token rejected; requesting a new token")
                token = await self._request_token()
                resp = await self.request(
                    "AuthenticationRequest",
                    {
                        "pluginName": self.config.plugin_name,
                        "pluginDeveloper": self.config.plugin_developer,
                        "authenticationToken": token,
                    },
                )
                data = resp.get("data") or {}
            if not data.get("authenticated"):
                raise VTSError(ERR_AUTH_REQUIRED, str(data.get("reason", "authentication failed")))
        self.authenticated = True
        self.log.info("VTS authenticated")

    async def _request_token(self) -> str:
        resp = await self.request(
            "AuthenticationTokenRequest",
            {
                "pluginName": self.config.plugin_name,
                "pluginDeveloper": self.config.plugin_developer,
            },
            timeout=180.0,
        )
        data = resp.get("data") or {}
        token = data.get("authenticationToken")
        if not isinstance(token, str) or not token:
            raise VTSMessageError("VTS did not return an authentication token")
        self._save_token(token)
        return token

    def _load_token(self) -> str | None:
        path = self.config.token_path
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except Exception as exc:
            self.log.warning("cannot read VTS token: %s", exc)
            return None
        token = data.get("authenticationToken") or data.get("token")
        if isinstance(token, str) and token:
            return token
        return None

    def _save_token(self, token: str) -> None:
        path = self.config.token_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "authenticationToken": token,
                    "pluginName": self.config.plugin_name,
                    "pluginDeveloper": self.config.plugin_developer,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        self.log.info("saved VTS token to %s", path)

    # -- queries -----------------------------------------------------------

    async def api_state(self) -> dict[str, Any]:
        return await self.request("APIStateRequest")

    async def current_model(self) -> dict[str, Any]:
        return (await self.request("CurrentModelRequest")).get("data") or {}

    async def list_input_params(self) -> dict[str, Any]:
        return (await self.request("InputParameterListRequest")).get("data") or {}

    async def list_live2d_params(self) -> dict[str, Any]:
        return (await self.request("Live2DParameterListRequest")).get("data") or {}

    async def parameter_value(self, name: str) -> dict[str, Any]:
        return (
            await self.request("ParameterValueRequest", {"name": name})
        ).get("data") or {}

    async def statistics(self) -> dict[str, Any]:        return (await self.request("StatisticsRequest")).get("data") or {}

    # -- injection ---------------------------------------------------------

    async def inject(
        self,
        params: dict[str, float],
        *,
        mode: str = "set",
        face_found: bool = True,
        weight: float | None = None,
    ) -> None:
        """Send one parameter batch.

        This is intentionally fire-and-forget for the 30 Hz path: awaiting a
        response per frame would add latency and backpressure. APIErrors are
        collected by the receiver loop.
        """
        if not self.authenticated:
            raise VTSMessageError("VTS is not authenticated")
        values: list[dict[str, Any]] = []
        for key, raw in params.items():
            value = float(raw)
            if value != value or value in (float("inf"), float("-inf")):
                continue
            item: dict[str, Any] = {"id": key, "value": value}
            if mode == "set" and weight is not None:
                item["weight"] = max(0.0, min(1.0, float(weight)))
            values.append(item)
        if not values:
            return
        rid = _new_request_id()
        payload = {
            "apiName": API_NAME,
            "apiVersion": API_VERSION,
            "messageType": "InjectParameterDataRequest",
            "requestID": rid,
            "data": {
                "faceFound": bool(face_found),
                "mode": mode,
                "parameterValues": values,
            },
        }
        await self._send(payload)

    def drain_inject_errors(self) -> list[VTSError]:
        errors = self._inject_errors[:]
        self._inject_errors.clear()
        return errors

    # -- convenience -------------------------------------------------------

    @staticmethod
    def now_ms() -> float:
        return time.perf_counter() * 1000.0
