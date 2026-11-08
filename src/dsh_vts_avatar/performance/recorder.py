"""Read-only expression recording.

The user performs an expression in VTube Studio while this recorder polls the
input and Live2D parameter lists. It never writes a parameter. The result is a
compact delta profile that can be stored on the model profile and used as a
custom expression hint.
"""

from __future__ import annotations

import asyncio
import json
import math
import multiprocessing as mp
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2

from ..vts.client import VTSClient

NEUTRAL = {
    "MouthOpen": 0.0, "MouthSmile": 0.10, "JawOpen": 0.0,
    "MouthFunnel": 0.0, "MouthPucker": 0.0, "MouthShrug": 0.0,
    "MouthPressLipOpen": 0.0, "MouthX": 0.0, "CheekPuff": 0.0,
    "TongueOut": 0.0, "VoiceA": 0.0, "VoiceI": 0.0, "VoiceU": 0.0,
    "VoiceE": 0.0, "VoiceO": 0.0, "VoiceSilence": 1.0,
    "FaceAngleX": 0.0, "FaceAngleY": 0.0, "FaceAngleZ": 0.0,
    "EyeOpenLeft": 1.0, "EyeOpenRight": 1.0,
    "EyeLeftX": 0.0, "EyeLeftY": 0.0, "EyeRightX": 0.0, "EyeRightY": 0.0,
    "Brows": 0.10, "BrowInnerUp": 0.0, "BrowLeftY": 0.0, "BrowRightY": 0.0,
    "FaceAngry": 0.0,
}


def _values(raw: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in raw.get("defaultParameters", []) + raw.get("customParameters", []):
        name = item.get("name")
        if name:
            out[str(name)] = float(item.get("value", 0.0))
    return out


def _live_values(raw: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in raw.get("parameters", []):
        name = item.get("name")
        if name:
            out[str(name)] = float(item.get("value", 0.0))
    return out


class _AudioCapture:
    def __init__(self, path: Path, device: int | None = None):
        self.path = path
        self.device = device
        self.frames: list[bytes] = []
        self.stream = None

    def start(self) -> None:
        try:
            import sounddevice as sd

            self.stream = sd.InputStream(
                samplerate=48000,
                channels=1,
                dtype="int16",
                device=self.device,
                callback=self._callback,
            )
            self.stream.start()
        except Exception:
            self.stream = None

    def _callback(self, indata, frames, time_info, status) -> None:
        self.frames.append(bytes(indata))

    def stop(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        if self.frames:
            try:
                with wave.open(str(self.path), "wb") as fh:
                    fh.setnchannels(1)
                    fh.setsampwidth(2)
                    fh.setframerate(48000)
                    fh.writeframes(b"".join(self.frames))
            except Exception:
                pass


class _FrameCapture:
    """Capture NDI frames in a separate process.

    JPEG encoding is CPU-heavy; doing it in the same Python process fights the
    asyncio parameter poller for the GIL. A child process keeps 30fps capture
    and parameter polling independent.
    """

    def __init__(self, ndi: Any, frame_dir: Path, start_time: float):
        self.source_hint = str(getattr(ndi, "source_hint", "") or "")
        self.frame_dir = frame_dir
        self.start_time = start_time
        self.events_path = frame_dir.parent / "frames.events.jsonl"
        self._ctx = mp.get_context("spawn")
        self._stop = self._ctx.Event()
        self._proc = self._ctx.Process(
            target=_frame_worker,
            args=(
                self.source_hint,
                str(frame_dir),
                start_time,
                str(self.events_path),
                self._stop,
            ),
            name="ndi-capture",
            daemon=True,
        )
        self.fps = 30.0
        self.warmup_s = 1.0

    def start(self) -> None:
        self._proc.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        self._proc.join(timeout=5.0)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2.0)
        events: list[dict[str, Any]] = []
        if self.events_path.exists():
            for line in self.events_path.read_text(encoding="utf-8").splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return events


def _frame_worker(
    source_hint: str,
    frame_dir: str,
    start_time: float,
    events_path: str,
    stop_event: Any,
) -> None:
    import cv2

    from ..ndi.capture import NDICapture

    out_dir = Path(frame_dir)
    ndi = NDICapture(source_hint=source_hint, snapshot_dir=out_dir)
    events: list[dict[str, Any]] = []
    if not ndi.open(wait_ms=4000):
        Path(events_path).write_text("", encoding="utf-8")
        return
    count = 0
    started = time.perf_counter()
    next_at = started
    try:
        while not stop_event.is_set():
            try:
                frame, _ = ndi.capture(timeout_ms=120)
            except Exception:
                frame = None
            if frame is not None and time.perf_counter() - started >= 1.0:
                count += 1
                name = f"frame_{count:05d}.jpg"
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if ok:
                    (out_dir / name).write_bytes(encoded.tobytes())
                    events.append(
                        {
                            "t": round(time.perf_counter() - start_time, 4),
                            "file": f"frames/{name}",
                        }
                    )
            next_at += 1.0 / 15.0
            delay = next_at - time.perf_counter()
            if delay < 0:
                next_at = time.perf_counter()
                delay = 0
            time.sleep(delay)
    finally:
        ndi.close()
        Path(events_path).write_text(
            "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + ("\n" if events else ""),
            encoding="utf-8",
        )


async def _sample_input(client: VTSClient) -> dict[str, float]:
    return _values(await client.list_input_params())


async def _sample_live(client: VTSClient) -> dict[str, float]:
    return _live_values(await client.list_live2d_params())


async def _sample(client: VTSClient) -> tuple[dict[str, float], dict[str, float]]:
    inputs = await client.list_input_params()
    live = await client.list_live2d_params()
    return _values(inputs), _live_values(live)


async def record_expression(
    client: VTSClient,
    name: str,
    *,
    duration_s: float = 4.0,
    baseline_s: float = 0.8,
    hz: float = 10.0,
    live_every: int = 2,
    on_start: Callable[[], None] | None = None,
    on_done: Callable[[], None] | None = None,
    on_session_start: Callable[[], None] | None = None,
    ndi: Any | None = None,
    session_dir: Path | None = None,
    audio_device: int | None = None,
) -> dict[str, Any]:
    duration_s = max(1.0, min(120.0, float(duration_s)))
    hz = max(5.0, min(30.0, float(hz)))
    model = await client.current_model()
    period = 1.0 / hz

    session = Path(session_dir) if session_dir is not None else None
    frame_dir = None
    params_fh = None
    frames_fh = None
    audio = None
    frame_capture = None
    frame_count = 0
    if session is not None:
        session.mkdir(parents=True, exist_ok=True)
        frame_dir = session / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        params_fh = (session / "params.jsonl").open("w", encoding="utf-8")
        frames_fh = (session / "frames.jsonl").open("w", encoding="utf-8")
        if audio_device is not None:
            audio = _AudioCapture(session / "audio.wav", audio_device)
            audio.start()
    session_start = time.perf_counter()
    if on_session_start is not None:
        on_session_start()
    if params_fh is not None:
        params_fh.write(json.dumps({"t": 0.0, "phase": "session"}, ensure_ascii=False) + "\n")
    if ndi is not None and frame_dir is not None:
        frame_capture = _FrameCapture(ndi, frame_dir, session_start)
        frame_capture.start()

    def write_sample(phase: str, inputs: dict[str, float], live: dict[str, float]) -> None:
        if params_fh is not None:
            params_fh.write(
                json.dumps(
                    {
                        "t": round(time.perf_counter() - session_start, 4),
                        "phase": phase,
                        "input": inputs,
                        "live2d": live,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # VTS keeps the last injected value after a plugin stops writing. Send one
    # neutral frame so the baseline is not polluted by the previous expression,
    # then go read-only for the actual take.
    await client.inject(NEUTRAL)
    await asyncio.sleep(1.0)
    baseline_inputs: list[dict[str, float]] = []
    baseline_live: list[dict[str, float]] = []

    for index in range(max(1, math.ceil(baseline_s * hz))):
        i = await _sample_input(client)
        l = await _sample_live(client) if index % max(1, live_every) == 0 else {}
        baseline_inputs.append(i)
        if l:
            baseline_live.append(l)
        write_sample("baseline", i, l)
        await asyncio.sleep(period)

    def average(rows: list[dict[str, float]]) -> dict[str, float]:
        keys = set().union(*(row.keys() for row in rows)) if rows else set()
        return {
            key: sum(row.get(key, 0.0) for row in rows) / max(1, len(rows))
            for key in keys
        }

    base_in = average(baseline_inputs)
    base_live = average(baseline_live)
    cue_t = None
    if on_start is not None:
        on_start()
        cue_t = round(time.perf_counter() - session_start, 4)
        if params_fh is not None:
            params_fh.write(
                json.dumps({"t": cue_t, "phase": "cue"}, ensure_ascii=False) + "\n"
            )
    samples_inputs: list[dict[str, float]] = []
    samples_live: list[dict[str, float]] = []
    started = time.perf_counter()
    while time.perf_counter() - started < duration_s:
        index = len(samples_inputs)
        i = await _sample_input(client)
        l = await _sample_live(client) if index % max(1, live_every) == 0 else {}
        samples_inputs.append(i)
        if l:
            samples_live.append(l)
        write_sample("expression", i, l)
        await asyncio.sleep(period)
    if on_done is not None:
        on_done()
    done_t = round(time.perf_counter() - session_start, 4)
    frame_events = frame_capture.stop() if frame_capture is not None else []
    frame_count = len(frame_events)
    if frames_fh is not None:
        for event in frame_events:
            frames_fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def summarize(
        base: dict[str, float],
        samples: list[dict[str, float]],
        *,
        threshold: float = 0.015,
    ) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        keys = set(base) | set().union(*(row.keys() for row in samples)) if samples else set(base)
        for key in keys:
            values = [row.get(key, 0.0) for row in samples]
            if not values:
                continue
            peak = max(values, key=lambda value: abs(value - base.get(key, 0.0)))
            delta = peak - base.get(key, 0.0)
            if abs(delta) < threshold:
                continue
            out[key] = {
                "base": round(base.get(key, 0.0), 5),
                "peak": round(peak, 5),
                "delta": round(delta, 5),
                "min": round(min(values), 5),
                "max": round(max(values), 5),
            }
        return dict(sorted(out.items(), key=lambda kv: abs(kv[1]["delta"]), reverse=True))

    inputs = summarize(base_in, samples_inputs)
    live = summarize(base_live, samples_live, threshold=0.01)
    summary = {
        "name": name,
        "created": datetime.now(timezone.utc).isoformat(),
        "model_id": str(model.get("modelID", "")),
        "model_name": str(model.get("modelName", "")),
        "duration_s": round(duration_s, 2),
        "samples": len(samples_inputs),
        "live2d_samples": len(samples_live),
        "frame_fps": 15,
        "baseline_inputs": {k: round(v, 5) for k, v in base_in.items()},
        "input_deltas": inputs,
        "live2d_deltas": live,
        "top_inputs": list(inputs)[:12],
        "top_live2d": list(live)[:12],
        "cue_t": cue_t,
        "done_t": done_t,
    }
    if audio is not None:
        audio.stop()
    for fh in (params_fh, frames_fh):
        if fh is not None:
            fh.close()
    if session is not None:
        summary["session_dir"] = str(session)
        summary["frames"] = frame_count
        summary["audio"] = bool(audio and audio.frames)
        (session / "manifest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return summary

