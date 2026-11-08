"""The local avatar engine: TTS -> timeline -> audio clock -> mixer -> VTS."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from collections import deque
from pathlib import Path
from dataclasses import replace
from typing import Any, Awaitable, Callable

import numpy as np

from ..audio.engine import Clock, make_clock
from ..config import Config
from ..ndi.capture import NDICapture
from ..performance import PerformanceMixer, strip_emotion_tags
from ..performance.affect import AffectEvidence, AffectFusion, AffectLog, AffectState
from ..performance.emotion_memory import EmotionEpisode, EpisodeStore, Mood, LABEL_CN
from ..performance.relationship import RelationshipStore, DELTAS
from ..performance.privacy import PrivacyStore
from ..performance.diary import DiaryStore
from ..performance.mouth_monitor import MouthMonitor
from ..performance.presence import MOUTH_KINDS, PresenceDirector, idle_mouth_allowed
from .local_chat import LocalChat, LocalDirector
from ..performance.recorder import record_expression
from ..tts import PiperTTS, SyntheticTTS
from ..tts.distill import DistilledPiperTTS
from ..tts.rvc import RVCConverter
from ..tts.timeline import Timeline
from ..vts.client import VTSClient, VTSError, find_vts_port
from ..vts.profile import ModelProfile, load_for_model, profile_path
from .segmentation import Segmenter
from .text_filter import SpeechTextFilter

log = logging.getLogger(__name__)

EventSink = Callable[[dict[str, Any]], Awaitable[None]]


class AvatarEngine:
    def __init__(self, config: Config, *, on_event: EventSink | None = None):
        self.config = config
        self.on_event = on_event
        self.profile = ModelProfile.load(config.contract_path)
        self.tts: PiperTTS | SyntheticTTS | None = None
        self.clock: Clock | None = None
        self.mixer: PerformanceMixer | None = None
        self.vts = VTSClient(config.vts)
        self.segmenter = Segmenter()
        self.speech_filter = SpeechTextFilter()
        self.segmenter_gesture_rate = config.avatar.gesture_rate
        self.timeline: Timeline | None = None
        self._parts: list[Timeline] = []
        self._pending_gap_s = 0.0
        self._queue: asyncio.Queue[str] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._ticker: asyncio.Task[None] | None = None
        self._watcher: asyncio.Task[None] | None = None
        self._closing = False
        self._utterance_open = False
        self._last_emit = 0.0
        self._last_sample = 0
        self._started = False
        self.utterance_id = ""
        self._last_tts_ms = 0.0
        self._last_error = ""
        self.ndi: NDICapture | None = None
        self._last_ndi = None
        self._affect_buffer = ""
        self._model_id = self.profile.model_id
        self._last_model_check = 0.0
        self._model_refreshing = False
        self._recording = False
        self._last_params: dict[str, float] = {}
        self._speech_generation = 0
        self.mouth_monitor = MouthMonitor()
        self.rvc = RVCConverter(config.rvc)
        self._rng = random.Random()
        # Phase 1/3: append-only record of every evidence record and the state
        # it produced, so a session can be replayed and audited later.
        self.affect_log = AffectLog(config.root / "data" / "affect")
        # Phase 4: sources vote; nothing except a manual instruction outranks
        # another source. See 用户原话与共识.md for the requirement.
        self.fusion = AffectFusion()
        # Emotion memory (phase 1): one episode per felt moment, keeping why.
        self.episode_store = EpisodeStore(config.root / "data" / "emotion" / "episodes.jsonl")
        # L3: numbers stay in here; only the impression reaches the prompt.
        self.relationship = RelationshipStore(config.root / "data")
        self.privacy = PrivacyStore(config.root / "data")
        self.diary = DiaryStore(config.root / "data")
        self.user_id = "default"
        self.display_name = "default"
        self._last_voice_at = 0.0
        self.presence = PresenceDirector(
            base_energy=float(getattr(config.avatar, "idle_energy", 0.55)),
            library_path=str(config.root / "data" / "presence" / "idle_library.json"),
        )
        self.local_chat = LocalChat(config.llm)
        self.director = LocalDirector(config.llm) if config.llm.director_enabled else None
        self._chat_history: dict[str, deque] = {}

    def switch_user(self, user_id: str, display_name: str = "") -> dict[str, Any]:
        """Switch the active person; every L2/L3 store follows."""
        cleaned = "".join(ch for ch in str(user_id or "") if ch.isalnum() or ch in "-_")
        if not cleaned:
            return {"ok": False, "error": "user_id required"}
        self.user_id = cleaned
        self.display_name = (display_name or cleaned).strip()[:40] or cleaned
        data = self.config.root / "data"
        episode_path = (
            data / "emotion" / "episodes.jsonl"
            if cleaned == "default"
            else data / "emotion" / f"{cleaned}.jsonl"
        )
        self.episode_store = EpisodeStore(episode_path)
        self.relationship = RelationshipStore(data, cleaned)
        self.privacy = PrivacyStore(data, cleaned)
        self.diary = DiaryStore(data, cleaned)
        return {"ok": True, "user_id": cleaned, "display_name": self.display_name}

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._load_tts()
        await self._warmup_tts()
        self.clock = make_clock(self.config.audio, self.tts.sample_rate)
        self.mixer = PerformanceMixer(
            self.config.avatar,
            self.tts.sample_rate,
            profile=self.profile,
            backend=self.profile.choose_mouth_backend(self.config.avatar, "auto"),
            use_affect=self.config.emotion.is_affect,
        )
        self.mixer.set_state("idle")
        self._queue = asyncio.Queue(maxsize=16)
        self._worker = asyncio.create_task(self._synth_worker(), name="avatar-tts")
        await self._connect_vts()
        self._watcher = asyncio.create_task(self._vts_watchdog(), name="avatar-vts-startup-check")
        if self.config.audio.auto_calibrate and hasattr(self.clock, "calibrate_output_latency"):
            # Only measure when there is no usable cache. Re-running this on
            # every start plays three quiet clicks, which is audible and is not
            # worth it while the endpoint has not changed.
            if not self.clock.has_fresh_latency():
                await asyncio.to_thread(self.clock.calibrate_output_latency)
            else:
                log.info("audio latency cache is fresh; skipping calibration")
        self._open_ndi()
        self._ticker = asyncio.create_task(self._tick_loop(), name="avatar-tick")
        await self._emit(
            {
                "type": "ready",
                "vts_connected": self.vts.connected,
                "profile": self.profile.summary(),
                "tts": type(self.tts).__name__,
                "sample_rate": self.tts.sample_rate,
                "backend": self.mixer.backend if self.mixer else "none",
            }
        )

    def _load_tts(self) -> None:
        if self.config.tts.distill_model is not None:
            distilled = DistilledPiperTTS(self.config.tts)
            if distilled.available:
                try:
                    distilled.load()
                    self.tts = distilled
                    log.info("using distilled Piper TTS at %d Hz", distilled.sample_rate)
                    return
                except Exception as exc:
                    self._last_error = f"Distilled Piper unavailable: {exc}"
                    log.warning("%s; falling back to reference Piper", self._last_error)
        piper = PiperTTS(self.config.tts)
        if piper.available:
            try:
                piper.load()
                self.tts = piper
                log.info("using Piper TTS at %d Hz", piper.sample_rate)
                return
            except Exception as exc:
                self._last_error = f"Piper unavailable: {exc}"
                log.warning("%s; falling back to silent synthetic TTS", self._last_error)
        else:
            self._last_error = "Piper voice/config not found; using silent synthetic TTS"
            log.warning(self._last_error)
        self.tts = SyntheticTTS(language=self.config.tts.language, sample_rate=48000)

    async def _warmup_tts(self) -> None:
        """Pay the first ONNX inference cost before the first real reply."""
        if self.tts is None:
            return
        try:
            started = time.perf_counter()
            await asyncio.to_thread(self.tts.synthesize, "你好。")
            log.info("TTS warmup finished in %.0f ms", (time.perf_counter() - started) * 1000)
        except Exception as exc:
            log.warning("TTS warmup failed: %s", exc)

    async def _connect_vts(self) -> bool:
        try:
            await self.vts.connect()
        except Exception as exc:
            # A VTS restart can re-bind its API on a neighbouring port. Probe
            # before falling back to the error path; otherwise every restart
            # silently disconnects the avatar until the bridge is restarted.
            port = await find_vts_port(self.config.vts.host, exclude=self.config.vts.port)
            if port is None:
                return await self._vts_connect_failed(exc)
            log.info("VTS API port moved %s -> %s", self.config.vts.port, port)
            self.config.vts.port = port
            try:
                await self.vts.connect()
            except Exception as retry_exc:
                return await self._vts_connect_failed(retry_exc)
        try:
            self.profile = await self._query_profile()
        except Exception as exc:
            return await self._vts_connect_failed(exc)
        self._model_id = self.profile.model_id
        if self.mixer is not None:
            self.mixer.set_profile(
                self.profile,
                self.profile.choose_mouth_backend(self.config.avatar, "auto"),
            )
        log.info("VTS ready: %s", self.profile.summary())
        return True

    async def _vts_connect_failed(self, exc: Exception) -> bool:
        self._last_error = str(exc)
        log.warning("VTS connection failed: %s", exc)
        await self._emit(
            {
                "type": "error",
                "code": "vts_connect_failed",
                "message": str(exc),
                "retryable": True,
            }
        )
        return False

    async def _vts_watchdog(self) -> None:
        """One startup check, then leave the connection alone.

        Same policy as the audio calibration: probe once when the bridge
        starts (including a VTS API port that moved), and do not poll. A later
        VTS restart is re-attached by the explicit ``reconnect`` message.
        """
        await asyncio.sleep(3.0)
        if self._closing or self._recording:
            return
        if self.vts.connected and self.vts.authenticated:
            return
        log.info("VTS startup check: connection lost, reconnecting once")
        try:
            await self.vts.close()
        except Exception:
            pass
        try:
            if await self._connect_vts():
                self._last_error = ""
                log.info("VTS startup check: reconnected")
                await self._emit({"type": "vts.reconnected", "profile": self.profile.summary()})
        except Exception as exc:
            self._last_error = f"VTS reconnect failed: {exc}"
            log.warning(self._last_error)

    async def _query_profile(self) -> ModelProfile:
        """Build a model profile from the live VTS parameter lists.

        A contract probe is model-specific. If the saved contract belongs to a
        different model (the user switched avatars), do not apply its probe
        results; build a fresh capability profile instead.
        """
        model = await self.vts.current_model()
        model_id = str(model.get("modelID", ""))
        model_name = str(model.get("modelName", ""))
        inputs = await self.vts.list_input_params()
        live2d = await self.vts.list_live2d_params()
        saved = load_for_model(self.config.profile_dir, model_id)
        if not saved.input_params:
            # Migration fallback: the old single contract.json may hold this
            # model's tuning/probes. Only use it if the model ID matches.
            legacy = ModelProfile.load(self.config.contract_path)
            if legacy.model_id and legacy.model_id == model_id:
                saved = legacy
        if saved.input_params:
            saved.model_name = model_name or saved.model_name
            saved.mouth_backend = saved.choose_mouth_backend(self.config.avatar, self.config.avatar.mouth_backend)
            return saved
        profile = ModelProfile(model_name=model_name, model_id=model_id)
        from ..vts.profile import ParameterInfo

        for raw in inputs.get("defaultParameters", []):
            info = ParameterInfo.from_vts(raw)
            if info.name:
                profile.input_params[info.name] = info
        for raw in inputs.get("customParameters", []):
            info = ParameterInfo.from_vts(raw)
            if info.name:
                profile.input_params[info.name] = info
        for raw in live2d.get("parameters", []):
            info = ParameterInfo.from_vts(raw)
            if info.name:
                profile.live2d_params[info.name] = info
        profile.mouth_backend = profile.choose_mouth_backend(self.config.avatar, self.config.avatar.mouth_backend)
        return profile

    async def _refresh_model(self) -> None:
        if self._model_refreshing or not self.vts.authenticated:
            return
        self._model_refreshing = True
        try:
            model = await self.vts.current_model()
            model_id = str(model.get("modelID", ""))
            if not model_id or model_id == self._model_id:
                return
            log.info("VTS model changed: %s -> %s", self._model_id, model_id)
            await self.cancel()
            profile = await self._query_profile()
            self.profile = profile
            self._model_id = model_id
            if self.mixer is not None:
                self.mixer.set_profile(
                    profile,
                    profile.choose_mouth_backend(self.config.avatar, "auto"),
                )
            await self._emit(
                {
                    "type": "model.changed",
                    "model": profile.summary(),
                    "backend": profile.mouth_backend,
                }
            )
        except Exception as exc:
            log.debug("model refresh failed: %s", exc)
        finally:
            self._model_refreshing = False

    async def reconnect(self) -> bool:
        if self.vts.authenticated:
            return True
        return await self._connect_vts()

    def _open_ndi(self) -> None:
        try:
            capture = NDICapture(source_hint=self.config.ndi.source, snapshot_dir=self.config.snapshot_dir)
            if capture.open(wait_ms=1200):
                self.ndi = capture
                log.info("NDI self-view connected")
            else:
                capture.close()
                log.info("NDI self-view not found (enable NDI output in VTS)")
        except Exception as exc:
            log.info("NDI self-view unavailable: %s", exc)

    async def snapshot(self, frames: int = 1, interval_ms: int = 180) -> dict[str, Any] | None:
        """Capture one frame or a short NDI sequence for self-inspection."""
        if self.ndi is None:
            self._open_ndi()
        if self.ndi is None:
            return None
        try:
            if frames > 1:
                metrics_list, sheet = await asyncio.to_thread(
                    self.ndi.capture_sequence, frames, interval_ms
                )
                if not metrics_list or sheet is None:
                    return None
                last = metrics_list[-1]
                self._last_ndi = last
                return {
                    "path": sheet,
                    "sequence": sheet,
                    "paths": [item.snapshot for item in metrics_list],
                    "frames": len(metrics_list),
                    "width": last.width,
                    "height": last.height,
                    "brightness": round(last.brightness, 2),
                    "motion": round(max(item.motion for item in metrics_list), 3),
                    "mean_motion": round(
                        sum(item.motion for item in metrics_list) / len(metrics_list), 3
                    ),
                    "mouth_motion": round(max(item.mouth_motion for item in metrics_list), 3),
                    "mean_mouth_motion": round(
                        sum(item.mouth_motion for item in metrics_list) / len(metrics_list), 3
                    ),
                    "head_motion": round(max(item.head_motion for item in metrics_list), 3),
                    "mean_head_motion": round(
                        sum(item.head_motion for item in metrics_list) / len(metrics_list), 3
                    ),
                    "source": last.source,
                    "timestamp": last.timestamp,
                }
            metrics = await asyncio.to_thread(self.ndi.snapshot)
            if metrics is None:
                return None
            self._last_ndi = metrics
            return {
                "path": metrics.snapshot,
                "width": metrics.width,
                "height": metrics.height,
                "brightness": round(metrics.brightness, 2),
                "motion": round(metrics.motion, 3),
                "mouth_motion": round(metrics.mouth_motion, 3),
                "head_motion": round(metrics.head_motion, 3),
                "source": metrics.source,
                "timestamp": metrics.timestamp,
            }
        except Exception as exc:
            log.warning("NDI snapshot failed: %s", exc)
            return None

    async def close(self) -> None:
        self._closing = True
        # Give the model a short release ramp before dropping the VTS
        # connection; abruptly stopping injection makes VTS snap back to idle.
        if self.vts.authenticated and self.mixer is not None and self.clock is not None:
            try:
                self.mixer.set_state("idle")
                for _ in range(10):
                    params = self.mixer.compute(self.clock.now_sample(), self.timeline)
                    if params:
                        await self.vts.inject(params)
                    await asyncio.sleep(1.0 / 30.0)
            except Exception:
                pass
        for task in (self._ticker, self._worker, self._watcher):
            if task is not None:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        if self.clock is not None:
            self.clock.stop()
        if self.ndi is not None:
            self.ndi.close()
            self.ndi = None
        await self.vts.close()

    # -- public speech API -------------------------------------------------

    async def begin_utterance(self, utterance_id: str | None = None) -> None:
        await self.cancel()
        self._last_voice_at = time.perf_counter()
        self.utterance_id = utterance_id or f"utt-{int(time.time() * 1000)}"
        self.segmenter = Segmenter()
        self.speech_filter.reset()
        self._affect_buffer = ""
        self._segment_count = 0
        self._utterance_open = True
        if self.mixer is not None:
            self.mixer.set_state("speaking")

    async def feed_text(self, text: str) -> None:
        if not text:
            return
        self._last_voice_at = time.perf_counter()
        if self.mixer is not None:
            # Keywords can be split across deltas; evaluate a rolling window.
            # 2026-11-07: only the tool self-report drives affect now. Explicit
            # [emotion:...] tags are still stripped from speech, but no longer
            # create affect evidence; keyword matching is off entirely.
            self._affect_buffer = (self._affect_buffer + text)[-32:]
            affect = None
            if affect:
                self.mixer.set_emotion(*affect)
                if self.config.emotion.is_affect:
                    self._push_affect(affect[0], affect[1])
        clean = strip_emotion_tags(text)
        clean = self.speech_filter.feed(clean)
        if not clean:
            return
        segments = self.segmenter.feed(clean)
        for segment in segments:
            self._enqueue(segment)

    async def end_utterance(self) -> None:
        self._utterance_open = False
        tail = self.speech_filter.flush()
        if tail:
            for segment in self.segmenter.feed(tail):
                self._enqueue(segment)
        for segment in self.segmenter.flush():
            self._enqueue(segment)
        if self._queue is not None and self._queue.empty() and self.timeline is None:
            if self.mixer is not None:
                self.mixer.set_state("idle")
        # Diary is written by the model itself (vts_avatar_diary); the template
        # fallback stays available but no longer runs automatically.

    def diary_write(self, text: str) -> dict[str, Any]:
        """Let the model write today's diary itself (better than a template)."""
        diary = getattr(self, "diary", None)
        if diary is None:
            return {"ok": False, "error": "diary not ready"}
        body = (text or "").strip()
        if len(body) < 20:
            return {"ok": False, "error": "too short"}
        path = diary.write(body)
        return {"ok": True, "path": str(path) if path else ""}

    def _diary_narrative(self, episodes: list, now: float) -> str:
        """Turn episodes into a first-person diary, not a ledger."""
        hour_words = (
            (6, "凌晨"), (9, "早上"), (12, "上午"), (14, "中午"),
            (18, "下午"), (22, "晚上"), (24, "深夜"),
        )

        def when(ts: float) -> str:
            hour = time.localtime(ts).tm_hour
            for limit, word in hour_words:
                if hour < limit:
                    return word
            return "深夜"

        leads = ("有些", "有点", "挺", "其实挺")

        def feelings(ep) -> str:
            names = [
                LABEL_CN.get(name, name)
                for name, value in (ep.labels or {}).items()
                if value > 0.2
            ]
            if not names:
                return "心里说不上来是什么滋味"
            lead = leads[sum(ord(ch) for ch in (ep.what or "x")) % len(leads)]
            return "心里" + lead + "、".join(names)

        status_closing = {
            "accepted": "后来想通了，也就接受了。",
            "doubted": "他解释之后我信了一半，另一半还悬着。",
            "suppressed": "我没把这份情绪说出口，压着，也没真处理。",
            "open": "到现在还有点没消化完，就先记在这里吧。",
        }
        parts: list[str] = []
        first = episodes[0]
        what = (first.what or "").strip() or "发生了一件事"
        opening = f"今天{when(first.started_at)}，{what}。{feelings(first)}。"
        why = (first.why or "").strip()
        names = [LABEL_CN.get(n, n) for n, v in (first.labels or {}).items() if v > 0.2]
        if why and not any(name in why for name in names):
            opening += f"{why}，所以才会这样。"
        parts.append(opening)
        for ep in episodes[1:4]:
            text = (ep.what or "").strip()
            if not text:
                continue
            sentence = f"后来，{text}。{feelings(ep)}。"
            why = (ep.why or "").strip()
            names = [LABEL_CN.get(n, n) for n, v in (ep.labels or {}).items() if v > 0.2]
            if why and not any(name in why for name in names):
                sentence += f"{why}。"
            parts.append(sentence)
        closing = status_closing.get(episodes[-1].status, "")
        if closing:
            parts.append(closing)
        parts.append("——今天大概就是这样。")
        return "\n\n".join(parts)

    def _maybe_write_diary(self) -> None:
        """Fallback template recap — the model normally writes the diary itself."""
        store = getattr(self, "episode_store", None)
        diary = getattr(self, "diary", None)
        if store is None or diary is None:
            return
        try:
            if not diary.due():
                return
            now = time.time()
            episodes = [
                ep
                for ep in getattr(store, "_episodes", [])
                if (ep.started_at or 0) > 0 and now - ep.started_at < 86400 * 1.5
            ]
            if not episodes:
                return
            episodes.sort(key=lambda ep: ep.started_at or 0)
            body = self._diary_narrative(episodes[-4:], now)
            header = f"# {time.strftime('%Y-%m-%d', time.localtime(now))}"
            diary.write(header + "\n\n" + body)
        except Exception:  # never let a diary failure break speech
            return

    def _enqueue(self, segment: str) -> None:
        segment = segment.strip()
        if not segment or self._queue is None:
            return
        # Do not let a stray brace, fence or punctuation-only fragment reach
        # Piper; it produces no audio and used to poison the speech queue.
        if not any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in segment):
            return
        try:
            self._queue.put_nowait(segment)
        except asyncio.QueueFull:
            # Drop the oldest pending phrase rather than blocking text flow.
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Exception:
                pass
            try:
                self._queue.put_nowait(segment)
            except Exception:
                pass

    async def cancel(self) -> None:
        self._speech_generation += 1
        if self._queue is not None:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except Exception:
                    break
        self._parts.clear()
        self._pending_gap_s = 0.0
        self.timeline = None
        self._last_sample = 0
        self.mouth_monitor.reset()
        if self.clock is not None:
            self.clock.clear()
        if self.mixer is not None:
            self.mixer.set_state("idle")
            self.mixer.reset()
        self._utterance_open = False
        await self._emit({"type": "cancelled", "utterance_id": self.utterance_id})

    def set_state(self, state: str) -> None:
        if self.mixer is not None:
            self.mixer.set_state(state)

    def add_action(self, kind: str, intensity: float = 0.7) -> None:
        if self.mixer is not None:
            self.mixer.add_action(kind, intensity)

    def set_emotion(self, name: str, intensity: float = 0.7) -> None:
        if self.mixer is not None:
            self.mixer.set_emotion(name, intensity)
            if self.config.emotion.is_affect:
                # A manual instruction is a mid-weight vote (0.5), not a hard
                # override: freedom means no source can force the face.
                self._push_affect(name, intensity, source="manual", confidence=1.0)
            # Thinking/shy have no fixed standard take yet, but they must
            # still be triggerable. Improvise them from the generic preset
            # plus a combined head/eye/brow action.
            gesture = {
                "thinking": "think_side",
                "思考": "think_side",
                "shy": "shy_avoid",
                "害羞": "shy_avoid",
            }.get(name)
            if gesture and not any(cue.kind == gesture and not cue.finished for cue in self.mixer.actions):
                self.mixer.add_action(gesture, intensity)

    def _sample_expression_gain(self) -> float:
        """Natural 70% / exaggerated 30% (user answer 2026-11-06)."""
        share = max(0.0, min(1.0, self.config.avatar.exaggeration_share))
        if self._rng.random() < share:
            return max(1.0, self.config.avatar.exaggeration_gain)
        return 1.0

    def _push_affect(
        self,
        label: str,
        intensity: float,
        *,
        source: str | None = None,
        confidence: float | None = None,
    ) -> None:
        """Turn one legacy reading into one affect evidence record.

        Phase 2 stays single-source on purpose; Phase 4 is where several
        evidence records get weighed against each other.
        """
        if self.mixer is None or not self.config.emotion.is_affect:
            return
        if source is None:
            markers = ("[emotion", "[情绪", "【情绪")
            source = "tag" if any(marker in self._affect_buffer for marker in markers) else "keyword"
        if confidence is None:
            confidence = 0.9 if source == "tag" else 0.6
        evidence = AffectEvidence(
            source=source,
            labels={label: 1.0},
            intensity=intensity,
            confidence=confidence,
        )
        state = self.fusion.add(evidence)
        state.intensity = min(1.0, state.intensity * self._sample_expression_gain())
        self.affect_log.evidence(evidence)
        self.affect_log.state(state, applied=True)
        self.mixer.set_affect(state)
        # Keyword / tag hits become low-confidence episodes: the reason is only
        # the matched phrase, so `why` stays empty and the source is honest.
        snippet = self._affect_buffer.strip()
        if snippet:
            snippet = snippet[-24:]
        self._remember(
            what=snippet or label,
            labels=state.labels,
            intensity=state.intensity,
            confidence=confidence,
            source=source,
        )

    def _remember(
        self,
        *,
        what: str,
        why: str = "",
        appraisal: list[str] | None = None,
        labels: dict[str, float] | None = None,
        intensity: float = 0.0,
        confidence: float = 0.0,
        source: str = "tool",
        turn_ref: str = "",
    ) -> None:
        """Record one remembered moment (emotion memory phase 1)."""
        store = getattr(self, "episode_store", None)
        if store is None:
            return
        clean_labels = dict(labels or {})
        matters = any(
            share >= 0.2 and name not in ("neutral", "calm")
            for name, share in clean_labels.items()
        )
        if not what.strip() and not matters:
            return
        store.add(
            EmotionEpisode(
                what=what.strip(),
                why=why.strip(),
                appraisal=list(appraisal or []),
                labels=clean_labels,
                intensity=max(0.0, min(1.0, float(intensity))),
                confidence=max(0.0, min(1.0, float(confidence))),
                salience=max(0.15, min(1.0, float(intensity))),
                source=source,
                turn_ref=turn_ref,
            )
        )

    def explain(
        self,
        explanation: str = "",
        acceptance: float | None = None,
        status: str = "",
        defense: str | None = None,
        episode_id: str = "",
    ) -> dict[str, Any]:
        """Record how an explanation landed on the latest open episode.

        The model judges acceptance; the bridge only stores the verdict and
        applies the status to salience (accepted fades faster, suppressed holds).
        """
        store = getattr(self, "episode_store", None)
        if store is None:
            return {"ok": False, "error": "emotion store not ready"}
        episode = store.find(episode_id or None)
        if episode is None:
            return {"ok": False, "error": "no open episode to explain"}
        allowed = ("open", "doubted", "suppressed", "accepted")
        store.revise(
            episode,
            explanation=explanation.strip() or None,
            acceptance=acceptance,
            status=status if status in allowed else None,
            defense=defense.strip() if defense else None,
        )
        if episode.status == "accepted":
            self.relationship.apply("apology_accepted", note=explanation.strip())
        elif episode.status == "doubted":
            self.relationship.apply("apology_rejected", note=explanation.strip())
        return {
            "ok": True,
            "id": episode.id,
            "status": episode.status,
            "acceptance": round(episode.acceptance, 3),
            "mood_text": self.mood_payload()["text"],
        }

    def relationship_event(self, event: str, note: str = "") -> dict[str, Any]:
        """Record one relationship event the model judged as real."""
        rel = getattr(self, "relationship", None)
        if rel is None:
            return {"ok": False, "error": "relationship not ready"}
        if event not in DELTAS:
            return {"ok": False, "error": f"unknown event: {event}"}
        rel.apply(event, note=(note or "")[:120])
        return {"ok": True, "event": event}

    def private_action(
        self,
        action: str,
        content: str = "",
        kind: str = "feeling",
        privacy: float = 0.5,
        item_id: str = "",
    ) -> dict[str, Any]:
        store = getattr(self, "privacy", None)
        if store is None:
            return {"ok": False, "error": "privacy store not ready"}
        if action == "add":
            if not content.strip():
                return {"ok": False, "error": "content required"}
            item = store.add(kind, content, privacy)
            return {"ok": True, "id": item.id, "required": round(item.required, 3)}
        if action == "list":
            return {
                "ok": True,
                "items": [
                    {"id": it.id, "kind": it.kind, "privacy": it.privacy,
                     "required": round(it.required, 3), "content": it.content}
                    for it in store.unshared()
                ],
            }
        if action == "share":
            if not item_id:
                return {"ok": False, "error": "item_id required"}
            item = next((it for it in store.unshared() if it.id == item_id), None)
            if item is None:
                return {"ok": False, "error": "no such private item"}
            rel = getattr(self, "relationship", None)
            trust = rel.state.trust if rel is not None else 0.0
            if not store.can_share(item, trust):
                return {
                    "ok": False,
                    "error": "trust too low for this one",
                    "required": round(item.required, 3),
                    "trust": round(trust, 3),
                }
            store.mark_shared(item_id)
            if rel is not None:
                rel.apply("vulnerability_shared", note=item.content[:40])
            return {"ok": True, "id": item.id, "shared": True}
        return {"ok": False, "error": "unknown action"}

    def recall(self, query: str = "", count: int = 3) -> dict[str, Any]:
        """Recent diary pages plus matching episodes (the 'look it up' path)."""
        count = max(1, min(10, int(count or 3)))
        diary: list[str] = []
        if getattr(self, "diary", None) is not None:
            diary = self.diary.recent(count)
        episodes: list[dict[str, Any]] = []
        store = getattr(self, "episode_store", None)
        if store is not None:
            needle = (query or "").strip()
            for ep in reversed(store._episodes):  # noqa: SLF001 - same package
                if needle and needle not in ep.what and needle not in ep.why:
                    continue
                episodes.append({
                    "id": ep.id, "what": ep.what, "why": ep.why,
                    "status": ep.status, "labels": ep.labels,
                })
                if len(episodes) >= count:
                    break
        return {"ok": True, "diary": diary, "episodes": episodes}

    def _refresh_baseline(self) -> None:
        """Push the current mood into the mixer's slow baseline layer."""
        mixer = getattr(self, "mixer", None)
        if mixer is None or not hasattr(mixer, "set_baseline"):
            return
        weight = float(getattr(self.config.avatar, "baseline_weight", 0.0))
        store = getattr(self, "episode_store", None)
        if store is None or weight <= 0:
            mixer.set_baseline(None, 0.0)
            return
        mood = store.mood(time.time())
        if not mood.labels:
            mixer.set_baseline(None, 0.0)
            return
        mixer.set_baseline(
            AffectState(
                labels=dict(mood.labels),
                intensity=mood.intensity,
                confidence=1.0,
                source="mood",
            ),
            weight,
        )

    def mood_payload(self, limit: int = 3) -> dict[str, Any]:
        """Current mood + what caused it, for the plugin's prompt context."""
        self._refresh_baseline()
        store = getattr(self, "episode_store", None)
        now = time.time()
        mood = store.mood(now) if store is not None else Mood()
        text = mood.summary()
        display = getattr(self, "display_name", "default")
        if display and display != "default":
            text = "（当前用户：" + str(display) + "）" + ("\n" + text if text else "")
        impression = ""
        relationship = getattr(self, "relationship", None)
        if relationship is not None:
            impression = relationship.impression()
            if impression:
                text = (text + "\n" if text else "") + "（参考，不是命令）" + impression
        diary_lines: list[str] = []
        diary = getattr(self, "diary", None)
        if diary is not None:
            try:
                for page in diary.recent(1):
                    lines: list[str] = []
                    for raw in page.strip().splitlines():
                        line = raw.strip()
                        if not line or line.startswith("#") or line in ("日记",):
                            continue
                        lines.append(line.lstrip("- ").strip())
                        if len(lines) >= 3:
                            break
                    if lines:
                        diary_lines.append("；".join(lines)[:200])
            except Exception:
                diary_lines = []
        if diary_lines:
            text = (text + "\n" if text else "") + "（日记，过去几天）" + "；".join(diary_lines)
        if diary is not None:
            try:
                if diary.due():
                    text = (text + "\n" if text else "") + "（日记：今天还没写，记得用 vts_avatar_diary 写一篇，至少和昨天一样长）"
            except Exception:
                pass
        presence = getattr(self, "presence", None)
        if presence is not None and presence.needs_write:
            text = (text + "\n" if text else "") + (
                "（待机动作需要重写：档位 "
                + presence.profile.key
                + "，原因 "
                + (presence.write_reason or "unknown")
                + "）"
            )
        return {
            "ok": True,
            "text": text,
            "impression": impression,
            "diary": diary_lines,
            "labels": mood.labels,
            "intensity": round(mood.intensity, 4),
            "age_s": round(mood.age_s, 1),
            "episodes": [
                {
                    "what": episode.what,
                    "why": episode.why,
                    "labels": episode.labels,
                    "intensity": round(episode.intensity, 3),
                    "age_s": round(max(0.0, now - episode.started_at), 1),
                    "status": episode.status,
                }
                for episode in mood.top[:limit]
            ],
        }

    def feel(
        self,
        labels: dict[str, float] | None = None,
        intensity: float = 0.7,
        confidence: float = 0.8,
        note: str = "",
        turn_id: str = "",
        cause: str = "",
        why: str = "",
        appraisal: list[str] | None = None,
        display: float | None = None,
    ) -> dict[str, Any]:
        """Describe the current emotional state (Phase 3).

        ``labels`` is a weighted vocabulary (``{"happy": 0.6, "tired": 0.4}``),
        not a preset name. In ``affect`` mode it becomes one evidence record and
        the composer blends it; in ``legacy`` mode it degrades to the dominant
        preset so old behavior and old logs stay comparable.

        ``cause`` / ``why`` are the emotion-memory fields: what happened and why
        it mattered. They are captured here, at the moment of feeling — never
        reconstructed later.
        """
        evidence = AffectEvidence(
            source="tool",
            labels=dict(labels or {}),
            intensity=intensity,
            confidence=confidence,
            note=note,
            turn_id=turn_id,
        )
        # Affect mode: this is one vote among several. Confidence decides how
        # much it counts (see affect/fusion.py); nothing here forces a preset.
        if self.config.emotion.is_affect:
            state = self.fusion.add(evidence)
            state.intensity = min(1.0, state.intensity * self._sample_expression_gain())
        else:
            state = AffectState.from_evidence(evidence)
        self.affect_log.evidence(evidence)
        if self.mixer is None:
            self.affect_log.state(state, applied=False)
            return {"ok": False, "error": "mixer not ready", "labels": state.labels}

        if self.config.emotion.is_affect:
            self.affect_log.state(state, applied=True)
            applied = state
            if display is not None:
                # felt stays in the log; displayed drives the face (2026-11-07)
                applied = replace(
                    state,
                    intensity=max(0.0, min(1.0, state.intensity * max(0.0, min(1.0, float(display))))),
                )
            self.mixer.set_affect(applied)
            dominant, weight = state.dominant()[0]
            if weight >= 0.5:
                # Keep the learned-expression/profile path alive; the face
                # targets themselves come from the composer.
                self.mixer.set_emotion(dominant, state.intensity)
        else:
            dominant, _weight = state.dominant()[0]
            self.affect_log.state(state, applied=True, mode="legacy")
            self.mixer.set_emotion(dominant, state.intensity)

        if self.config.emotion.is_affect:
            self._remember(
                what=cause,
                why=why,
                appraisal=appraisal,
                labels=state.labels,
                intensity=state.intensity,
                confidence=confidence,
                source="tool",
                turn_ref=turn_id,
            )

        # 松了口气: play the sigh cue (round-ish mouth + a breath out + head sink).
        top_label, top_weight = state.dominant()[0]
        if self.mixer is not None and top_label == "relieved" and top_weight >= 0.5:
            now = time.time()
            if now - getattr(self, "_last_sigh", 0.0) > 3.0:
                self.mixer.add_action("sigh", state.intensity)
                self._last_sigh = now

        return {
            "ok": True,
            "mode": self.config.emotion.mode,
            "labels": state.labels,
            "dominant": state.dominant(2),
            "intensity": round(state.intensity, 4),
            "confidence": state.confidence,
            "valence": round(state.valence, 4),
            "arousal": round(state.arousal, 4),
            "mood_text": self.mood_payload()["text"],
        }

    async def record_expression(self, name: str, seconds: float = 4.0) -> dict[str, Any]:
        """Read-only capture of a user-performed expression.

        The 30 Hz writer is paused for the whole capture; the user can move
        VTS with the mouse, face tracking, or VTS actions without our writes
        fighting them. The resulting delta profile is saved on this model.
        """
        if not self.vts.authenticated:
            raise RuntimeError("VTS is not connected")
        self._recording = True
        try:
            summary = await record_expression(self.vts, name, duration_s=seconds)
            if self.profile is not None:
                self.profile.learned_expressions[name] = summary
                direct = {
                    key: float(info["peak"])
                    for key, info in summary.get("input_deltas", {}).items()
                    if abs(float(info.get("delta", 0.0))) >= 0.02
                }
                self.profile.expression_map[name] = direct
                self.profile.save(profile_path(self.config.profile_dir, self.profile.model_id))
                if self.mixer is not None:
                    self.mixer.set_profile(self.profile)
            return summary
        finally:
            self._recording = False

    def demo_head(self, *, curve: str = "minjerk", intensity: float = 0.5, duration_ms: int = 900) -> bool:
        if self.clock is None or self.mixer is None or self.tts is None:
            return False
        at = self.clock.now_sample() + int(0.10 * self.tts.sample_rate)
        self.mixer.add_action(
            "demo_nod",
            intensity=max(0.05, min(1.4, intensity)),
            at_sample=at,
            duration_ms=duration_ms,
            curve="bezier" if curve == "bezier" else "minjerk",
        )
        return True

    # -- synthesis ---------------------------------------------------------

    async def _synth_worker(self) -> None:
        assert self._queue is not None
        while not self._closing:
            text = await self._queue.get()
            generation = self._speech_generation
            try:
                started = time.perf_counter()
                assert self.tts is not None
                timeline = await asyncio.to_thread(self.tts.synthesize, text)
                if self.rvc.enabled:
                    timeline = await asyncio.to_thread(self.rvc.convert, timeline, text)
                if generation != self._speech_generation:
                    # The user cancelled/began another utterance while this
                    # synthesis was in flight. Never append stale speech.
                    log.info("discarding stale TTS result")
                    continue
                self._last_tts_ms = (time.perf_counter() - started) * 1000.0
                self._append_timeline(timeline, text)
            except Exception as exc:
                self._last_error = f"TTS failed: {exc}"
                log.exception("TTS failed for %r", text)
                await self._emit(
                    {
                        "type": "error",
                        "code": "tts_failed",
                        "message": str(exc),
                        "retryable": True,
                    }
                )
            finally:
                self._queue.task_done()

    def _append_timeline(self, timeline: Timeline, text: str) -> None:
        if self.clock is None or self.mixer is None:
            return
        if self._parts and self._pending_gap_s > 0:
            sample_rate = int(timeline.sample_rate)
            gap_audio = np.zeros(int(sample_rate * self._pending_gap_s), dtype=np.float32)
            gap = Timeline(
                sample_rate=sample_rate,
                audio=gap_audio,
                spans=[],
                segments=[],
                rms=np.zeros(max(1, gap_audio.size // max(1, int(sample_rate * 0.01))), dtype=np.float32),
                rms_hop=max(1, int(sample_rate * 0.01)),
            )
            self.clock.append(gap.audio)
            self._parts.append(gap)
            self._pending_gap_s = 0.0
        start_sample = sum(part.duration_samples for part in self._parts)
        self._parts.append(timeline)
        self.timeline = Timeline.concat(self._parts)
        self.clock.append(timeline.audio)
        if not self.clock.is_running:
            self.clock.start()
        self.mixer.set_state("speaking")
        self._pending_gap_s = 0.0
        self._segment_count += 1
        # A small speech emphasis / gesture on selected phrase starts.
        if text and self.segmenter_gesture_rate > 0:
            sample_rate = self.tts.sample_rate if self.tts else 48000
            # If TTS fell behind playback, the scheduled start is already in
            # the past. Starting a cue mid-curve is exactly what looks like a
            # sudden head pop; clamp the start to the next tick instead.
            now_sample = self.clock.now_sample()
            start = max(
                start_sample + int(0.04 * sample_rate),
                now_sample + int(0.05 * sample_rate),
            )
            if "！" in text or "!" in text or "？" in text or "?" in text:
                self.mixer.add_action("emphasis", 0.50, at_sample=start)
            elif self._segment_count % 3 == 0:
                self.mixer.add_action("nod", 0.35, at_sample=start)
            elif self._segment_count % 4 == 0:
                # A slow look to the side at a phrase boundary, with the eyes
                # leading the head. This is a small cue, not a constant scan.
                side = "turn_look_left" if (self._segment_count // 4) % 2 else "turn_look_right"
                self.mixer.add_action(side, 0.28, at_sample=start)

    # -- control loop ------------------------------------------------------

    def _presence_inputs(self) -> dict[str, Any]:
        """Everything the idle director needs, read fresh each tick."""
        store = getattr(self, "episode_store", None)
        now = time.time()
        mood = store.mood(now) if store is not None else None
        rel = getattr(self, "relationship", None)
        state = getattr(rel, "state", None)
        diary = getattr(self, "diary", None)
        try:
            has_diary = bool(diary.recent(1)) if diary is not None else False
        except Exception:
            has_diary = False
        return {
            "mood_labels": dict(getattr(mood, "labels", {}) or {}),
            "mood_intensity": float(getattr(mood, "intensity", 0.0) or 0.0),
            "trust": float(getattr(state, "trust", 0.5) or 0.5),
            "closeness": float(getattr(state, "closeness", 0.2) or 0.2),
            "friction": float(getattr(state, "friction", 0.0) or 0.0),
            "dependency": float(getattr(state, "dependency", 0.0) or 0.0),
            "has_diary": has_diary,
        }

    async def local_reply(self, user_text: str, show_text: bool = False) -> dict[str, Any]:
        """Answer through the local model and speak it; never return the text."""
        cfg = self.config.llm
        if not cfg.enabled:
            return {"ok": False, "error": "local chat disabled"}
        text_in = (user_text or "").strip()
        if not text_in:
            return {"ok": False, "error": "text required"}
        history = self._chat_history.setdefault(
            self.user_id, deque(maxlen=max(4, int(cfg.history_turns) * 2))
        )
        extra = ""
        try:
            extra = self.mood_payload().get("text") or ""
        except Exception:
            extra = ""
        try:
            reply = await self.local_chat.reply(text_in, list(history), extra_system=extra)
        except Exception as exc:
            return {"ok": False, "error": f"llm: {exc}"}
        if not reply:
            return {"ok": False, "error": "empty reply"}
        history.append({"role": "user", "content": text_in})
        history.append({"role": "assistant", "content": reply})
        try:
            await self.begin_utterance()
            await self.feed_text(reply)
            await self.end_utterance()
        except Exception as exc:
            return {"ok": False, "error": f"tts: {exc}", "chars": len(reply)}
        self._log_local_turn(text_in, reply)
        result = {"ok": True, "chars": len(reply), "model": cfg.model}
        if show_text:
            # Only the local UI asks for this; the host agent never does.
            result["reply"] = reply
        return result

    def llm_settings(self) -> dict[str, Any]:
        cfg = self.config.llm
        return {
            "ok": True,
            "enabled": cfg.enabled,
            "backend": cfg.backend,
            "endpoint": cfg.endpoint,
            "model": cfg.model,
            "director_enabled": cfg.director_enabled,
            "director_model": cfg.director_model,
        }

    def save_llm_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        from ..tools.panel import set_section_value

        allowed = {
            "enabled", "backend", "endpoint", "api_key", "model",
            "director_enabled", "director_model",
        }
        path = self.config.root / "config.toml"
        try:
            text = path.read_text(encoding="utf-8")
            for key, value in values.items():
                if key not in allowed:
                    continue
                text = set_section_value(text, "llm", key, value)
            path.write_text(text, encoding="utf-8")
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "restart_required": True}

    def settings_snapshot(self) -> dict[str, Any]:
        a, au, llm = self.config.avatar, self.config.audio, self.config.llm
        return {
            "ok": True,
            "llm": {
                "enabled": llm.enabled, "backend": llm.backend,
                "endpoint": llm.endpoint, "model": llm.model,
                "director_enabled": llm.director_enabled,
                "director_model": llm.director_model,
            },
            "avatar": {
                "head_gain": a.head_gain, "mouth_gain": a.mouth_gain,
                "expression_gain": getattr(a, "expression_gain", 1.0),
                "baseline_weight": a.baseline_weight, "blink_per_min": a.blink_per_min,
            },
            "audio": {
                "device": au.device, "auto_calibrate": au.auto_calibrate,
                "calibration_volume": au.calibration_volume,
                "latency_ms": self.clock.auto_latency_ms() if hasattr(self.clock, "auto_latency_ms") else None,
            },
            "persona": llm.persona,
            "user_id": self.user_id,
        }

    def save_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        from ..tools.panel import set_section_value
        path = self.config.root / "config.toml"
        try:
            text = path.read_text(encoding="utf-8")
            for section in ("llm", "avatar", "audio"):
                block = values.get(section)
                if isinstance(block, dict):
                    for key, value in block.items():
                        text = set_section_value(text, section, key, value)
            persona = values.get("persona")
            if isinstance(persona, str) and persona.strip():
                text = set_section_value(text, "llm", "persona", persona.strip())
            path.write_text(text, encoding="utf-8")
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "restart_required": True}

    def open_diary_folder(self) -> dict[str, Any]:
        import os
        import subprocess
        folder = self.config.root / "data" / "diary" / self.user_id
        try:
            folder.mkdir(parents=True, exist_ok=True)
            subprocess.Popen(["explorer", str(folder)], creationflags=0x08000000 if os.name == "nt" else 0)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "path": str(folder)}

    def request_idle_write(self) -> dict[str, Any]:
        self.presence.needs_write = True
        self.presence.write_reason = "manual"
        return {"ok": True, "profile": self.presence.profile.key}

    async def recalibrate_audio(self) -> dict[str, Any]:
        import asyncio
        try:
            value = await asyncio.to_thread(self.clock.calibrate_output_latency)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": value is not None, "latency_ms": value}

    def open_panel(self) -> dict[str, Any]:
        """Open the local settings panel (writes config.toml, never the host)."""
        import os
        import subprocess
        import sys

        script = Path(__file__).resolve().parents[1] / "tools" / "panel.py"
        if not script.exists():
            return {"ok": False, "error": "panel script missing"}
        config = self.config.root / "config.toml"
        flags = 0x08000000 if os.name == "nt" else 0
        try:
            subprocess.Popen(
                [sys.executable, str(script), "--config", str(config)],
                creationflags=flags,
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "config": str(config)}

    def open_transcript_window(self) -> dict[str, Any]:
        """Open the local transcript viewer (no content passes through the host)."""
        import os
        import subprocess
        import sys

        script = Path(__file__).resolve().parents[1] / "tools" / "chat_viewer.py"
        if not script.exists():
            return {"ok": False, "error": "viewer script missing"}
        chat = self.config.root / "data" / "chat" / f"{self.user_id}.jsonl"
        flags = 0x08000000 if os.name == "nt" else 0
        try:
            subprocess.Popen(
                [sys.executable, str(script), "--file", str(chat)],
                creationflags=flags,
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "file": str(chat)}

    def _log_local_turn(self, user_text: str, reply: str) -> None:
        """Keep the raw transcript on disk; it is never sent through the host."""
        try:
            path = self.config.root / "data" / "chat" / f"{self.user_id}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "user": user_text,
                "assistant": reply,
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            log.debug("local transcript write failed: %s", exc)

    def idle_write(self, profile: str, actions: list, note: str = "") -> dict[str, Any]:
        """Store a model-written idle batch (tool path)."""
        try:
            self.presence.save_batch(profile, actions, note)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "profile": profile, "actions": len(actions or [])}

    def _director_context(self) -> dict[str, Any]:
        """Everything the avatar already knows, in a compact form for the director."""
        inputs = self._presence_inputs()
        context: dict[str, Any] = {
            "profile": self.presence.profile.key,
            "mood": inputs.get("mood_labels") or {},
            "mood_intensity": self.presence.profile.energy,
            "trust": inputs.get("trust"),
            "closeness": inputs.get("closeness"),
            "friction": inputs.get("friction"),
            "has_diary": inputs.get("has_diary"),
            "baseline_weight": getattr(self.config.avatar, "baseline_weight", 0.4),
            "recent": self.presence.recent(),
        }
        try:
            rel = getattr(self, "relationship", None)
            impression = rel.impression() if rel is not None else ""
            if impression:
                context["relation"] = str(impression)[:60]
        except Exception:
            pass
        try:
            diary = getattr(self, "diary", None)
            if diary is not None:
                recent = diary.recent(1)
                if recent:
                    context["diary"] = str(recent[0])[:80]
        except Exception:
            pass
        return context

    async def _maybe_idle_cue(self, now: float) -> None:
        """Ask the presence director for the next idle motion, if any."""
        if self.mixer is None or self.mixer.state != "idle":
            return
        self.presence.check_expiry(time.time())
        if self.mixer.actions:
            return  # something is already playing; never stack idle cues
        self.presence.observe(self._presence_inputs())
        silence = (
            time.perf_counter() - self._last_voice_at if self._last_voice_at else 0.0
        )
        rule = self.presence.next_cue(now, speaking=False, silence_s=silence)
        if rule is None:
            return
        kind, intensity = rule
        mouth_ok = idle_mouth_allowed(self.mixer.state, self.timeline is not None)
        if self.director is not None:
            kinds = [
                name
                for name in self.presence.pool_kinds()
                if name not in MOUTH_KINDS or mouth_ok
            ]
            try:
                suggestion = await asyncio.wait_for(
                    self.director.suggest(
                        kinds,
                        {
                            **self._director_context(),
                            "silence_s": round(silence, 1),
                        },
                    ),
                    timeout=self.config.llm.director_timeout_s + 1.0,
                )
            except Exception:
                suggestion = None
            if suggestion is not None:
                kind, intensity = suggestion
                self.presence.note(kind)
                log.info("director picked: %s (%.2f)", kind, intensity)
        if kind in MOUTH_KINDS and not mouth_ok:
            return
        self.mixer.add_action(kind, intensity)
        log.info(
            "idle cue: %s (%.2f) profile=%s",
            kind,
            intensity,
            self.presence.profile.key,
        )

    async def _tick_loop(self) -> None:
        period = 1.0 / 30.0
        next_at = asyncio.get_running_loop().time()
        while not self._closing:
            now = time.perf_counter()
            if self._recording:
                # Record-only mode: never write while the user is performing.
                await asyncio.sleep(period)
                continue
            if self.vts.authenticated and now - self._last_model_check >= 2.0:
                self._last_model_check = now
                asyncio.create_task(self._refresh_model(), name="avatar-model-refresh")
            if self.clock is not None and self.mixer is not None:
                if self.timeline is None:
                    # Idle: the audio clock is stopped, so advance our own
                    # sample counter from wall time — otherwise every action
                    # cue is frozen at sample 0 and never animates.
                    now_s = time.perf_counter()
                    if not hasattr(self, "_idle_origin") or self._idle_origin is None:
                        self._idle_origin = now_s
                    sample = int((now_s - self._idle_origin) * self.mixer.sample_rate)
                else:
                    sample = self.clock.now_sample()
                self._last_sample = sample
                params = self.mixer.compute(sample, self.timeline, now)
                self._last_params = params
                if self.timeline is not None:
                    mouth_segment = self.timeline.at(sample)
                    audible = self.timeline.rms_at(sample) > 0.01 or (
                        mouth_segment.kind != "silence" and mouth_segment.open > 0.02
                    )
                else:
                    audible = False
                self.mouth_monitor.observe(
                    now,
                    params,
                    speaking=self.mixer.state == "speaking",
                    audible=audible,
                )
                if self.vts.authenticated and params:
                    try:
                        await self.vts.inject(params)
                    except Exception as exc:
                        if not self._last_error:
                            self._last_error = str(exc)
                        log.debug("inject failed: %s", exc)
                for err in self.vts.drain_inject_errors():
                    self._last_error = f"VTS {err.error_id}: {err.vts_message}"
                    log.warning("VTS injection error: %s", err)
                if self._should_idle(sample):
                    self.mixer.set_state("idle")
                await self._maybe_idle_cue(now)
                if now - self._last_emit >= 0.10:
                    self._last_emit = now
                    await self._emit(
                        {
                            "type": "playback",
                            "utterance_id": self.utterance_id,
                            "sample": sample,
                            "duration": self.timeline.duration_samples if self.timeline else 0,
                            "state": self.mixer.state,
                            "vts_connected": self.vts.connected,
                            "last_error": self._last_error,
                        }
                    )
            next_at += period
            delay = next_at - asyncio.get_running_loop().time()
            if delay < 0:
                # Do not accumulate a burst of catch-up frames.
                next_at = asyncio.get_running_loop().time()
                delay = 0
            await asyncio.sleep(delay)

    def _should_idle(self, sample: int) -> bool:
        if self._utterance_open:
            return False
        if self._queue is not None and not self._queue.empty():
            return False
        if self.timeline is None:
            return True
        return sample >= self.timeline.duration_samples

    # -- events ------------------------------------------------------------

    async def _emit(self, payload: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            await self.on_event(payload)
        except Exception:
            log.debug("event sink failed", exc_info=True)

    def parameters(self) -> dict[str, Any]:
        segment = self.timeline.at(self._last_sample) if self.timeline is not None else None
        return {
            "sample": self._last_sample,
            "state": self.mixer.state if self.mixer else "idle",
            "params": dict(self._last_params),
            "phoneme": segment.phoneme if segment else None,
            "viseme": segment.viseme if segment else None,
            "mouth_stuck": self.mouth_monitor.stuck,
            "mouth_activity": round(self.mouth_monitor.activity, 4),
        }

    def set_head_pose(self, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0, seconds: float = 8.0) -> bool:
        if self.mixer is None:
            return False
        self.mixer.set_head_pose(yaw, pitch, roll, seconds)
        return True

    def status(self) -> dict[str, Any]:
        return {
            "vts_connected": self.vts.connected,
            "vts_authenticated": self.vts.authenticated,
            "tts": type(self.tts).__name__ if self.tts else None,
            "backend": self.mixer.backend if self.mixer else None,
            "model": self.profile.model_name,
            "state": self.mixer.state if self.mixer else "idle",
            "sample": self._last_sample,
            "last_error": self._last_error,
            "mouth_stuck": self.mouth_monitor.stuck,
            "mouth_activity": round(self.mouth_monitor.activity, 4),
            "ndi_connected": self.ndi is not None,
            "last_ndi": self._last_ndi.__dict__ if self._last_ndi is not None else None,
            "contract": str(self.config.contract_path),
        }
