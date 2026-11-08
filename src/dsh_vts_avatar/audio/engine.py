"""Audio playback and the audible clock.

The mixer never trusts wall-clock time while speech is playing. It asks this
engine where the *audible* sample is, then looks up the corresponding phoneme.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from ..config import AudioConfig

log = logging.getLogger(__name__)

try:  # Optional high-quality resampler.
    import soxr  # type: ignore
except Exception:  # pragma: no cover - fallback path
    soxr = None


class Clock(Protocol):
    sample_rate: int

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def clear(self) -> None: ...
    def append(self, audio: np.ndarray) -> None: ...
    def now_sample(self) -> int: ...
    def duration_samples(self) -> int: ...
    @property
    def is_running(self) -> bool: ...


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src_rate == dst_rate or audio.size == 0:
        return audio
    if soxr is not None:
        return np.asarray(soxr.resample(audio, src_rate, dst_rate), dtype=np.float32)
    # Linear fallback; adequate for a first run, not for final quality.
    duration = audio.size / src_rate
    out_size = max(1, int(round(duration * dst_rate)))
    src_x = np.linspace(0.0, duration, audio.size, endpoint=False)
    dst_x = np.linspace(0.0, duration, out_size, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


class AudioEngine:
    """PortAudio output stream plus an audible sample counter."""

    def __init__(self, config: AudioConfig, source_rate: int, *, logger: logging.Logger | None = None):
        import sounddevice as sd

        self._sd = sd
        self.config = config
        self.source_rate = int(source_rate)
        self.sample_rate = int(config.sample_rate)
        self.log = logger or log
        self._lock = threading.Lock()
        self._audio = np.zeros(0, dtype=np.float32)
        self._cursor = 0
        self._frames_consumed = 0
        self._stream = None
        self._running = False
        self._latency_samples = 0
        self._auto_latency_ms = 0.0
        self._last_underrun_log = 0.0

    def _device_name(self, device: int | None) -> str:
        """Resolve a device name without assuming query_devices returns a dict.

        sounddevice returns a DeviceList when called without a concrete device;
        that used to crash at the first TTS chunk when the default output was
        selected dynamically.
        """
        try:
            info = (
                self._sd.query_devices(device)
                if device is not None
                else self._sd.query_devices(kind="output")
            )
            if isinstance(info, dict):
                return str(info.get("name", ""))
        except Exception:
            pass
        return ""

    def _pick_device(self) -> int | None:
        if self.config.device is not None:
            try:
                info = self._sd.query_devices(self.config.device)
                if int(info.get("max_output_channels", 0)) > 0:
                    return int(self.config.device)
                self.log.warning(
                    "configured audio device %s has no output channels; falling back to default",
                    self.config.device,
                )
            except Exception:
                self.log.warning(
                    "configured audio device %s is unavailable; falling back to default",
                    self.config.device,
                )
        try:
            # Prefer the WASAPI default when "default" is requested. PortAudio's
            # global default resolves through MME, which on this machine labels
            # the 3.5mm jack "耳机" but actually routes to the speakers.
            for host in self._sd.query_hostapis():
                if "wasapi" in str(host.get("name", "")).lower():
                    wasapi_out = int(host.get("default_output_device", -1))
                    if wasapi_out >= 0:
                        name = self._device_name(wasapi_out)
                        if any(token in name.lower() for token in ("bluetooth", "wh-1000", "hands-free")):
                            self.log.warning(
                                "default output looks like Bluetooth (%s); use wired audio for low latency",
                                name,
                            )
                        return wasapi_out
        except Exception:
            pass
        try:
            default = self._sd.default.device
            # sounddevice returns an _InputOutputPair here, not a tuple/list.
            # The pair itself has [0]=input and [1]=output but is not a list.
            out = default[1] if hasattr(default, "__getitem__") else default
            if out is not None and int(out) >= 0:
                out = int(out)
                name = self._device_name(out)
                if any(token in name.lower() for token in ("bluetooth", "wh-1000", "hands-free")):
                    self.log.warning(
                        "default output looks like Bluetooth (%s); use wired audio for low latency",
                        name,
                    )
                return int(out)
        except Exception:
            pass
        return None

    def start(self) -> None:
        if self._running:
            return
        device = self._pick_device()
        self._stream = self._sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=256,
            device=device,
            latency="low",
            callback=self._callback,
        )
        self._stream.start()
        self._running = True
        try:
            latency_s = float(self._stream.latency)
        except Exception:
            latency_s = 0.02
        self._latency_samples = int(latency_s * self.sample_rate) + int(
            self.config.av_offset_ms * self.sample_rate / 1000.0
        )
        self.log.info(
            "audio started: device=%s name=%r rate=%d latency=%.1fms av_offset=%dms",
            device,
            self._device_name(device),
            self.sample_rate,
            latency_s * 1000,
            self.config.av_offset_ms,
        )

    def calibrate_output_latency(self) -> float | None:
        """Measure this output endpoint once with a quiet click + loopback.

        This replaces the old hard-coded 70ms estimate. It runs after VTS
        authentication, before the first utterance, and falls back to the
        driver-reported latency if loopback is unavailable.
        """
        try:
            import soundcard as sc
        except Exception as exc:
            self.log.warning("auto latency calibration unavailable: %s", exc)
            return None
        try:
            speaker = sc.default_speaker()
            mic = sc.get_microphone(speaker.name, include_loopback=True)
        except Exception as exc:
            self.log.warning("auto latency calibration: no loopback: %s", exc)
            return None

        # Warm the output stream first. Starting the stream inside the timed
        # click adds device startup latency and overestimates the steady-state
        # playback latency.
        if not self.is_running:
            self.start()
            time.sleep(0.25)

        recorded: dict[str, object] = {}

        def record() -> None:
            import ctypes

            try:
                ctypes.windll.ole32.CoInitializeEx(None, 0x2)
            except Exception:
                pass
            try:
                with mic.recorder(samplerate=self.sample_rate, channels=1) as recorder:
                    recorded["start"] = time.perf_counter()
                    recorded["audio"] = np.asarray(
                        recorder.record(numframes=int(self.sample_rate * 1.2))
                    ).reshape(-1)
            finally:
                try:
                    ctypes.windll.ole32.CoUninitialize()
                except Exception:
                    pass

        delays: list[float] = []
        for attempt in range(3):
            recorded.clear()
            thread = threading.Thread(target=record, name=f"latency-loopback-{attempt}")
            thread.start()
            time.sleep(0.12)
            self.clear()
            click = np.zeros(int(self.source_rate * 0.05), dtype=np.float32)
            tone = np.sin(2 * np.pi * 1000 * np.arange(int(self.source_rate * 0.03)) / self.source_rate)
            click[: tone.size] = (tone * self.config.calibration_volume).astype(np.float32)
            t0 = time.perf_counter()
            self.append(click)
            if not self.is_running:
                self.start()
            thread.join()
            audio = np.asarray(recorded.get("audio", np.zeros(0)), dtype=np.float32)
            record_start = float(recorded.get("start", t0))
            self.clear()
            if audio.size == 0 or float(np.max(np.abs(audio))) < 0.005:
                self.log.warning("auto latency calibration attempt %d: no click", attempt + 1)
                continue
            index = int(np.argmax(np.abs(audio)))
            delay_ms = (record_start + index / self.sample_rate - t0) * 1000.0
            if 15.0 <= delay_ms <= 500.0:
                delays.append(delay_ms)
            else:
                self.log.warning(
                    "auto latency calibration attempt %d rejected: %.1fms", attempt + 1, delay_ms
                )
        if not delays:
            cached = self._read_cached_latency()
            if cached is not None:
                self._auto_latency_ms = cached
                self._latency_samples = int(cached * self.sample_rate / 1000.0)
                self.log.warning("auto latency calibration failed; using cached %.1fms", cached)
                return cached
            self.log.warning("auto latency calibration failed; keeping driver estimate")
            return None
        if len(delays) < 2 or max(delays) - min(delays) > 30.0:
            cached = self._read_cached_latency()
            if cached is not None:
                self._auto_latency_ms = cached
                self._latency_samples = int(cached * self.sample_rate / 1000.0)
                self.log.warning("auto latency calibration unstable; using cached %.1fms", cached)
                return cached
            self.log.warning("auto latency calibration unstable; keeping driver estimate")
            return None
        delay_ms = float(np.median(delays)) + float(self.config.sync_correction_ms)
        self._auto_latency_ms = delay_ms
        self._latency_samples = int(delay_ms * self.sample_rate / 1000.0)
        self._write_cached_latency(delay_ms)
        self.log.info(
            "auto latency calibration: %.1fms (median of %d + %.1fms correction, click volume %.2f, replaces driver estimate)",
            delay_ms,
            len(delays),
            self.config.sync_correction_ms,
            self.config.calibration_volume,
        )
        return delay_ms

    def has_fresh_latency(self) -> bool:
        """True when a usable cached latency exists, so calibration can be skipped."""
        return self._read_cached_latency() is not None

    @staticmethod
    def _latency_cache_path() -> Path:
        return Path(__file__).resolve().parents[3] / "data" / "audio_latency.json"

    def _read_cached_latency(self) -> float | None:
        try:
            data = json.loads(self._latency_cache_path().read_text(encoding="utf-8"))
            value = float(data.get("latency_ms"))
            if 15.0 <= value <= 500.0:
                driver_ms = 0.0
                if self._stream is not None:
                    driver_ms = float(getattr(self._stream, "latency", 0.0)) * 1000.0
                if driver_ms > 0 and abs(value - driver_ms) > 150.0:
                    self.log.warning(
                        "cached latency %.1fms disagrees with driver (%.1fms); ignoring cache",
                        value,
                        driver_ms,
                    )
                    return None
                return value
        except Exception:
            pass
        return None

    def _write_cached_latency(self, value: float) -> None:
        try:
            path = self._latency_cache_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "latency_ms": round(float(value), 1),
                        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            self.log.warning("could not cache latency: %s", exc)

    def _callback(self, outdata, frames, _time_info, status) -> None:
        if status:
            # Avoid log spam during underruns.
            now = time.perf_counter()
            if now - self._last_underrun_log > 1.0:
                self._last_underrun_log = now
                self.log.warning("audio callback status: %s", status)
        with self._lock:
            end = self._cursor + frames
            chunk = self._audio[self._cursor:end]
            if chunk.size < frames:
                outdata[: chunk.size, 0] = chunk
                outdata[chunk.size :, 0] = 0.0
            else:
                outdata[:, 0] = chunk
            self._cursor = min(end, self._audio.size)
            self._frames_consumed += frames

    def append(self, audio: np.ndarray) -> None:
        data = _resample(audio, self.source_rate, self.sample_rate)
        # Short fades at chunk edges: without them the jump from silence to
        # waveform (and the cut back to silence at the end) is an audible click.
        fade = min(int(self.sample_rate * 0.004), max(1, data.size // 4))
        if fade > 1 and data.size > fade * 2:
            data = data.copy()
            data[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            data[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        with self._lock:
            # A little silence between chunks avoids clicks at boundaries.
            self._audio = np.concatenate([self._audio, data])
            self._audio = self._audio.astype(np.float32, copy=False)

    def clear(self) -> None:
        with self._lock:
            self._audio = np.zeros(0, dtype=np.float32)
            self._cursor = 0
            self._frames_consumed = 0

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def now_sample(self) -> int:
        with self._lock:
            played = max(0, self._frames_consumed - self._latency_samples)
        # Map audible output samples back to the source timeline.
        if self.sample_rate == self.source_rate:
            return int(played)
        return int(played * self.source_rate / self.sample_rate)

    def duration_samples(self) -> int:
        with self._lock:
            return int(self._audio.size * self.source_rate / self.sample_rate)

    @property
    def is_running(self) -> bool:
        return self._running


class NullAudioEngine:
    """Silent clock for VTS-only testing.

    Appending audio advances the virtual timeline by its duration; playback
    then follows monotonic time. This exercises the exact same mixer path
    without requiring a working output device.
    """

    def __init__(self, config: AudioConfig, source_rate: int):
        self.config = config
        self.source_rate = int(source_rate)
        self.sample_rate = int(config.sample_rate)
        self._total = np.zeros(0, dtype=np.float32)
        self._started_at: float | None = None
        self._offset = 0

    def start(self) -> None:
        if self._started_at is None:
            self._started_at = time.perf_counter()

    def stop(self) -> None:
        self._started_at = None
        self._total = np.zeros(0, dtype=np.float32)
        self._offset = 0

    def clear(self) -> None:
        self._total = np.zeros(0, dtype=np.float32)
        self._offset = 0
        self._started_at = time.perf_counter()

    def append(self, audio: np.ndarray) -> None:
        data = _resample(audio, self.source_rate, self.sample_rate)
        self._total = np.concatenate([self._total, data])

    def now_sample(self) -> int:
        if self._started_at is None:
            return 0
        elapsed = time.perf_counter() - self._started_at
        played = int(elapsed * self.sample_rate) + self._offset
        played = min(played, int(self._total.size))
        if self.sample_rate == self.source_rate:
            return played
        return int(played * self.source_rate / self.sample_rate)

    def duration_samples(self) -> int:
        return int(self._total.size * self.source_rate / self.sample_rate)

    @property
    def is_running(self) -> bool:
        return self._started_at is not None


def make_clock(config: AudioConfig, source_rate: int) -> Clock:
    if config.enabled:
        return AudioEngine(config, source_rate)
    return NullAudioEngine(config, source_rate)
