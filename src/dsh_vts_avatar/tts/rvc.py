"""Optional Feiyu RVC voice conversion.

The Piper timeline stays the timing master. Only the waveform is replaced, so
lip sync and phoneme boundaries do not move.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
import urllib.request
import uuid
import wave

import numpy as np

from ..config import RVCConfig
from .timeline import Timeline


class RVCConverter:
    def __init__(self, config: RVCConfig, *, logger: logging.Logger | None = None):
        self.config = config
        self.log = logger or logging.getLogger(__name__)

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def _cache_key(self, text: str, timeline: Timeline) -> str:
        raw = "\n".join(
            [
                "rvc-v1",
                self.config.voice,
                text,
                str(timeline.sample_rate),
                str(timeline.audio.size),
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _key(self) -> str:
        try:
            key = self.config.api_key_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(f"cannot read Feiyu API key: {exc}") from exc
        if not key:
            raise RuntimeError("Feiyu API key is empty")
        return key

    def _request(self, method: str, path: str, *, data: bytes | None = None,
                 content_type: str | None = None) -> bytes:
        headers = {"Authorization": "Bearer " + self._key()}
        if content_type:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(
            self.config.base_url.rstrip("/") + path,
            data=data,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            return response.read()

    def _upload(self, wav: bytes) -> str:
        boundary = "----dsh-rvc-" + uuid.uuid4().hex
        head = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="input.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8")
        tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
        body = head + wav + tail
        raw = self._request(
            "POST",
            "/v1/assets",
            data=body,
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        return str(json.loads(raw)["id"])

    def _convert(self, asset_id: str) -> bytes:
        payload = json.dumps(
            {"voice": self.config.voice, "audio_id": asset_id}, ensure_ascii=False
        ).encode("utf-8")
        raw = self._request(
            "POST",
            "/v1/voice-conversion",
            data=payload,
            content_type="application/json; charset=utf-8",
        )
        job_id = str(json.loads(raw)["id"])
        deadline = time.monotonic() + max(30.0, self.config.timeout_s)
        last = None
        while time.monotonic() < deadline:
            job = json.loads(self._request("GET", f"/v1/jobs/{job_id}"))
            status = str(job.get("status", ""))
            if status != last:
                self.log.info("RVC job %s -> %s", job_id[:8], status)
                last = status
            if status == "success":
                return self._request("GET", f"/v1/jobs/{job_id}/audio")
            if status in {"failed", "interrupted"}:
                raise RuntimeError(f"RVC job ended {status}: {job.get('error', 'no detail')}")
            time.sleep(1.0)
        raise TimeoutError(f"RVC job {job_id} timed out after {self.config.timeout_s}s")

    @staticmethod
    def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate or samples.size == 0:
            return samples.astype(np.float32, copy=False)
        target_size = max(1, int(round(samples.size * target_rate / source_rate)))
        old_t = np.arange(samples.size, dtype=np.float64) / source_rate
        new_t = np.arange(target_size, dtype=np.float64) / target_rate
        return np.interp(new_t, old_t, samples).astype(np.float32)

    @staticmethod
    def _fit(samples: np.ndarray, size: int) -> np.ndarray:
        if samples.size >= size:
            return samples[:size].astype(np.float32, copy=False)
        return np.pad(samples, (0, size - samples.size)).astype(np.float32, copy=False)

    @staticmethod
    def _decode_wav(data: bytes) -> tuple[np.ndarray, int]:
        import io

        with wave.open(io.BytesIO(data), "rb") as fh:
            rate = fh.getframerate()
            channels = fh.getnchannels()
            width = fh.getsampwidth()
            raw = fh.readframes(fh.getnframes())
        if width == 2:
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 4:
            samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise RuntimeError(f"unsupported RVC WAV sample width: {width}")
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        return samples, rate

    @staticmethod
    def _write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
        with wave.open(str(path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(rate)
            fh.writeframes(pcm.tobytes())

    @staticmethod
    def _read_wav(path: Path) -> tuple[np.ndarray, int]:
        with wave.open(str(path), "rb") as fh:
            rate = fh.getframerate()
            channels = fh.getnchannels()
            samples = np.frombuffer(
                fh.readframes(fh.getnframes()), dtype="<i2"
            ).astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        return samples, rate

    def convert(self, timeline: Timeline, text: str) -> Timeline:
        if not self.enabled:
            return timeline
        try:
            key = self._cache_key(text, timeline)
            cache = self.config.cache_dir / f"{key}.wav"
            if cache.is_file():
                samples, rate = self._read_wav(cache)
                samples = self._fit(self._resample(samples, rate, timeline.sample_rate), timeline.audio.size)
                return self._replace_audio(timeline, samples)
            tmp = self.config.cache_dir / f".input-{key}.wav"
            self._write_wav(tmp, timeline.audio, timeline.sample_rate)
            wav = tmp.read_bytes()
            tmp.unlink(missing_ok=True)
            asset_id = self._upload(wav)
            converted = self._convert(asset_id)
            samples, rate = self._decode_wav(converted)
            samples = self._fit(self._resample(samples, rate, timeline.sample_rate), timeline.audio.size)
            self._write_wav(cache, samples, timeline.sample_rate)
            return self._replace_audio(timeline, samples)
        except Exception as exc:
            self.log.warning("RVC conversion failed; using Piper audio: %s", exc)
            return timeline

    def _replace_audio(self, timeline: Timeline, samples: np.ndarray) -> Timeline:
        hop = max(1, int(timeline.sample_rate * 0.01))
        count = max(1, samples.size // hop)
        frames = samples[: count * hop].reshape(count, hop)
        rms = np.sqrt(np.mean(frames * frames, axis=1)).astype(np.float32)
        return Timeline(
            sample_rate=timeline.sample_rate,
            audio=samples.astype(np.float32, copy=False),
            spans=timeline.spans,
            segments=timeline.segments,
            rms=rms,
            rms_hop=hop,
            language=timeline.language,
        )
