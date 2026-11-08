"""NDI 5/6 receiver for the avatar self-view.

VTS can publish its model as an NDI source. We use that feed for two things:
1. a current JPEG the agent can inspect (``vts_ndi_snapshot``), and
2. continuous motion/energy telemetry so the performance layer can notice
   "the model is not moving" or "the mouth is stuck".
"""

from __future__ import annotations

from dataclasses import dataclass, field
import ctypes
import logging
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class NDIMetrics:
    timestamp: float
    width: int
    height: int
    brightness: float
    motion: float
    source: str
    snapshot: str | None = None
    mouth_motion: float = 0.0
    head_motion: float = 0.0


@dataclass
class NDICapture:
    source_hint: str = ""
    snapshot_dir: Path = Path("data/ndi")
    _ndi: object | None = None
    _find: object | None = None
    _recv: object | None = None
    _source: object | None = None
    _prev_gray: np.ndarray | None = None
    _frame_count: int = 0
    _last_metrics: NDIMetrics | None = None
    _connected: bool = False

    def open(self, *, wait_ms: int = 5000) -> bool:
        import NDIlib as ndi

        if not ndi.initialize():
            raise RuntimeError("NDI initialize() failed; is the NDI runtime installed?")
        self._ndi = ndi
        self._find = ndi.find_create_v2()
        if self._find is None:
            raise RuntimeError("NDI find_create_v2() failed")
        sources = self._wait_for_sources(wait_ms)
        if not sources:
            self.close()
            return False
        source = self._choose_source(sources)
        recv_create = ndi.RecvCreateV3()
        recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        self._recv = ndi.recv_create_v3(recv_create)
        if self._recv is None:
            self.close()
            raise RuntimeError("NDI recv_create_v3() failed")
        ndi.recv_connect(self._recv, source)
        self._source = source
        self._connected = True
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        log.info("NDI connected to %s", getattr(source, "ndi_name", "?"))
        return True

    def _wait_for_sources(self, wait_ms: int):
        import NDIlib as ndi

        assert self._find is not None
        deadline = time.monotonic() + wait_ms / 1000.0
        while time.monotonic() < deadline:
            ndi.find_wait_for_sources(self._find, 500)
            sources = ndi.find_get_current_sources(self._find)
            if sources:
                return list(sources)
        return list(ndi.find_get_current_sources(self._find) or [])

    def _choose_source(self, sources):
        if self.source_hint:
            needle = self.source_hint.lower()
            for source in sources:
                if needle in str(getattr(source, "ndi_name", "")).lower():
                    return source
        for source in sources:
            if "vtube" in str(getattr(source, "ndi_name", "")).lower():
                return source
        return sources[0]

    def capture(self, *, timeout_ms: int = 1500) -> tuple[np.ndarray | None, NDIMetrics | None]:
        import NDIlib as ndi

        if not self._connected or self._recv is None:
            return None, None
        deadline = time.monotonic() + timeout_ms / 1000.0
        video = None
        while time.monotonic() < deadline:
            t, candidate, audio, _meta = ndi.recv_capture_v2(
                self._recv, max(50, int((deadline - time.monotonic()) * 1000)), want_metadata=False
            )
            if t == ndi.FRAME_TYPE_AUDIO and audio is not None:
                ndi.recv_free_audio_v2(self._recv, audio)
                continue
            if t == getattr(ndi, "FRAME_TYPE_SOURCE_CHANGE", object()):
                # The receiver connects asynchronously; wait for the first video frame.
                continue
            if t == ndi.FRAME_TYPE_ERROR:
                self._connected = False
                return None, None
            if t == ndi.FRAME_TYPE_VIDEO and candidate is not None:
                video = candidate
                break
        if video is None:
            return None, None
        try:
            frame = self._to_bgr(video)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion = 0.0
            if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
                motion = float(np.mean(cv2.absdiff(gray, self._prev_gray)))
            self._prev_gray = gray
            self._frame_count += 1
            brightness = float(np.mean(gray))
            path = self.snapshot_dir / "latest.jpg"
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            metrics = NDIMetrics(
                timestamp=time.time(),
                width=int(video.xres),
                height=int(video.yres),
                brightness=brightness,
                motion=motion,
                source=str(getattr(self._source, "ndi_name", "")),
                snapshot=str(path),
            )
            self._last_metrics = metrics
            return frame, metrics
        finally:
            ndi.recv_free_video_v2(self._recv, video)

    @staticmethod
    def _to_bgr(video) -> np.ndarray:
        data = video.data
        if isinstance(data, np.ndarray):
            arr = data
        else:
            arr = np.ctypeslib.as_array(
                (ctypes.c_uint8 * (video.yres * video.xres * 4)).from_address(int(data))
            )
        arr = arr.reshape((video.yres, video.xres, 4))
        # BGRX -> BGR
        return np.ascontiguousarray(arr[:, :, :3])

    def snapshot(self) -> NDIMetrics | None:
        _frame, metrics = self.capture(timeout_ms=1000)
        return metrics or self._last_metrics

    def capture_sequence(self, count: int = 4, interval_ms: int = 180) -> tuple[list[NDIMetrics], str | None]:
        """Capture several frames and a contact sheet for motion inspection."""
        count = max(1, min(12, int(count)))
        interval = max(0, min(2000, int(interval_ms)))
        frames: list[np.ndarray] = []
        metrics_list: list[NDIMetrics] = []
        previous_gray: np.ndarray | None = None
        for index in range(count):
            frame, metrics = self.capture(timeout_ms=1500)
            if frame is None or metrics is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous_gray is not None and previous_gray.shape == gray.shape:
                metrics.mouth_motion = self._region_motion(previous_gray, gray, 0.40, 0.70, 0.34, 0.66)
                metrics.head_motion = self._region_motion(previous_gray, gray, 0.12, 0.52, 0.24, 0.76)
            previous_gray = gray
            path = self.snapshot_dir / f"sequence_{index:02d}.jpg"
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            metrics.snapshot = str(path)
            frames.append(frame)
            metrics_list.append(metrics)
            if index + 1 < count and interval:
                time.sleep(interval / 1000.0)
        if not frames:
            return [], None
        tiles = []
        height, width = frames[0].shape[:2]
        scale = min(1.0, 480.0 / max(1, width))
        tile_w, tile_h = max(1, int(width * scale)), max(1, int(height * scale))
        for index, frame in enumerate(frames):
            tile = cv2.resize(frame, (tile_w, tile_h))
            cv2.putText(tile, str(index), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            tiles.append(tile)
        columns = min(4, len(tiles))
        rows = []
        for start in range(0, len(tiles), columns):
            row = tiles[start:start + columns]
            while len(row) < columns:
                row.append(np.zeros_like(tiles[0]))
            rows.append(np.hstack(row))
        sheet = np.vstack(rows)
        sheet_path = self.snapshot_dir / "sequence.jpg"
        cv2.imwrite(str(sheet_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        self._last_metrics = metrics_list[-1]
        return metrics_list, str(sheet_path)

    @staticmethod
    def _region_motion(previous: np.ndarray, current: np.ndarray, y0: float, y1: float, x0: float, x1: float) -> float:
        height, width = current.shape[:2]
        ya, yb = max(0, int(height * y0)), min(height, int(height * y1))
        xa, xb = max(0, int(width * x0)), min(width, int(width * x1))
        if yb <= ya or xb <= xa:
            return 0.0
        return float(np.mean(cv2.absdiff(previous[ya:yb, xa:xb], current[ya:yb, xa:xb])))

    @property
    def metrics(self) -> NDIMetrics | None:
        return self._last_metrics

    def close(self) -> None:
        import NDIlib as ndi

        self._connected = False
        if self._recv is not None:
            try:
                ndi.recv_destroy(self._recv)
            except Exception:
                pass
            self._recv = None
        if self._find is not None:
            try:
                ndi.find_destroy(self._find)
            except Exception:
                pass
            self._find = None
        if self._ndi is not None:
            try:
                ndi.destroy()
            except Exception:
                pass
            self._ndi = None


def list_sources(wait_ms: int = 3000) -> list[dict[str, str]]:
    import NDIlib as ndi

    if not ndi.initialize():
        return []
    finder = ndi.find_create_v2()
    try:
        deadline = time.monotonic() + wait_ms / 1000.0
        while time.monotonic() < deadline:
            ndi.find_wait_for_sources(finder, 500)
            sources = ndi.find_get_current_sources(finder)
            if sources:
                return [
                    {
                        "name": str(getattr(source, "ndi_name", "")),
                        "url": str(getattr(source, "url_address", "")),
                    }
                    for source in sources
                ]
        return []
    finally:
        try:
            ndi.find_destroy(finder)
        except Exception:
            pass
        try:
            ndi.destroy()
        except Exception:
            pass


class NDIMonitor:
    """Background capture that keeps the latest frame and metrics."""

    def __init__(
        self,
        capture: NDICapture,
        *,
        interval_s: float = 0.5,
        on_metrics: Callable[[NDIMetrics], None] | None = None,
    ):
        self.capture = capture
        self.interval_s = interval_s
        self.on_metrics = on_metrics
        self._stop = False
        self._thread = None

    def start(self) -> None:
        import threading

        self._stop = False
        self._thread = threading.Thread(target=self._run, name="ndi-monitor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            try:
                _frame, metrics = self.capture.capture(timeout_ms=1000)
                if metrics and self.on_metrics:
                    self.on_metrics(metrics)
            except Exception:
                log.debug("NDI capture failed", exc_info=True)
            time.sleep(self.interval_s)

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
