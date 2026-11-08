# Author: 2722144646
# Created: 2026-11-07
# License: AGPL-3.0 (see LICENSE)
"""L2 emotion memory: episodes with causes, and a slow mood.

L1 answers "what does the face show right now". L2 answers "why did it feel
that way, and is it still with it". An episode keeps the cause, not just a
label, so the reason survives context clearing.

Rules that come from the plan (情绪记忆-初步计划.md):

- causes are captured at the moment of feeling, never reconstructed later;
- only the tool self-report feeds episodes (keyword/tag sources are off);
- salience half-life is 30-60 minutes (default 40);
- ``status`` / ``acceptance`` / ``defense`` are carried, but the acceptance
  transitions land in phase 2;
- nothing here writes to VTS.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
import uuid

from .affect.schema import normalize_labels

SCHEMA_VERSION = 2
DEFAULT_HALF_LIFE_S = 40 * 60.0  # 30-60 min per plan; 40 is the midpoint
MAX_AGE_S = 6 * 3600.0
MERGE_WINDOW_S = 8.0

LABEL_CN = {
    "happy": "开心", "joy": "开心", "excited": "兴奋",
    "sad": "难过", "angry": "生气", "surprised": "惊讶",
    "thinking": "思索", "curious": "好奇", "shy": "害羞",
    "wronged": "委屈", "tired": "累", "calm": "平静",
    "neutral": "平静", "suppressed": "压着", "doubtful": "半信半疑",
    "guilty": "心虚", "lonely": "孤单", "relieved": "松了口气",
}


@dataclass
class EmotionEpisode:
    """One thing that happened and how it landed."""

    what: str = ""
    why: str = ""
    appraisal: list[str] = field(default_factory=list)
    labels: dict[str, float] = field(default_factory=dict)
    intensity: float = 0.0
    confidence: float = 0.0
    salience: float = 0.5
    status: str = "open"
    acceptance: float = 0.0
    explanation: str | None = None
    defense: str | None = None
    source: str = "tool"
    user_id: str = "default"
    turn_ref: str = ""
    started_at: float = 0.0
    updated_at: float = 0.0
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"ep-{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "EmotionEpisode":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in payload.items() if key in known})


class EpisodeStore:
    """Append-only JSONL store with an in-memory index and salience decay."""

    def __init__(
        self,
        path: Path,
        *,
        clock=time.time,
        half_life_s: float = DEFAULT_HALF_LIFE_S,
        merge_window_s: float = MERGE_WINDOW_S,
    ) -> None:
        self.path = Path(path)
        self.clock = clock
        self.half_life_s = half_life_s
        self.merge_window_s = merge_window_s
        self._episodes: list[EmotionEpisode] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("schema_version", SCHEMA_VERSION) > SCHEMA_VERSION:
                continue
            try:
                self._episodes.append(EmotionEpisode.from_dict(payload))
            except TypeError:
                continue
        # Later revisions share the episode id and win.
        deduped: dict[str, EmotionEpisode] = {}
        for ep in self._episodes:
            deduped[ep.id] = ep
        self._episodes = list(deduped.values())

    def _append(self, episode: EmotionEpisode) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")
        except OSError:
            pass  # memory stays usable when the disk is not

    def add(self, episode: EmotionEpisode) -> EmotionEpisode:
        now = self.clock()
        if episode.started_at <= 0:
            episode.started_at = now
        if episode.updated_at <= 0:
            episode.updated_at = now
        last = self._episodes[-1] if self._episodes else None
        if (
            last is not None
            and last.status == "open"
            and last.source == episode.source
            and last.user_id == episode.user_id
            and abs(now - last.updated_at) <= self.merge_window_s
            and last.what == episode.what
        ):
            last.labels = _merge_labels(last.labels, episode.labels)
            last.intensity = max(last.intensity, episode.intensity)
            last.updated_at = now
            return last
        self._episodes.append(episode)
        self._append(episode)
        return episode

    def salience(self, episode: EmotionEpisode, now: float | None = None) -> float:
        now = self.clock() if now is None else now
        age = max(0.0, now - episode.started_at)
        half_life = self.half_life_s
        if episode.status == "accepted":
            half_life = max(1.0, half_life / 3.0)  # healing fades faster than time
        decay = 0.5 ** (age / max(1.0, half_life))
        if episode.status == "suppressed":
            decay = max(decay, 0.35)  # pressed down, not resolved
        return max(0.0, episode.salience * decay)

    def find(self, episode_id: str | None = None) -> EmotionEpisode | None:
        if episode_id:
            return next((ep for ep in reversed(self._episodes) if ep.id == episode_id), None)
        return next((ep for ep in reversed(self._episodes) if ep.status == "open"), None)

    def revise(
        self,
        episode: EmotionEpisode,
        *,
        explanation: str | None = None,
        acceptance: float | None = None,
        status: str | None = None,
        defense: str | None = None,
    ) -> EmotionEpisode:
        if explanation is not None:
            episode.explanation = explanation
        if acceptance is not None:
            episode.acceptance = max(0.0, min(1.0, acceptance))
        if status is not None:
            episode.status = status
        if defense is not None:
            episode.defense = defense
        episode.updated_at = self.clock()
        self._append(episode)
        return episode

    def active(
        self,
        now: float | None = None,
        limit: int = 20,
        user_id: str | None = None,
    ) -> list[EmotionEpisode]:
        now = self.clock() if now is None else now
        live = [
            ep
            for ep in self._episodes
            if now - ep.started_at <= MAX_AGE_S
            and self.salience(ep, now) > 0.02
            and (user_id is None or ep.user_id == user_id)
        ]
        live.sort(key=lambda ep: self.salience(ep, now), reverse=True)
        return live[:limit]

    def mood(self, now: float | None = None, user_id: str | None = None) -> "Mood":
        now = self.clock() if now is None else now
        episodes = self.active(now, user_id=user_id)
        if not episodes:
            return Mood()
        total = 0.0
        blended: dict[str, float] = {}
        intensity = 0.0
        for ep in episodes:
            weight = self.salience(ep, now)
            if weight <= 0:
                continue
            total += weight
            intensity += weight * ep.intensity
            for label, share in ep.labels.items():
                blended[label] = blended.get(label, 0.0) + weight * share
        if total <= 0:
            return Mood()
        started = min(ep.started_at for ep in episodes)
        return Mood(
            labels=normalize_labels(blended),
            intensity=min(1.0, intensity / total),
            started_at=started,
            age_s=max(0.0, now - started),
            top=episodes[:3],
        )


@dataclass
class Mood:
    labels: dict[str, float] = field(default_factory=dict)
    intensity: float = 0.0
    started_at: float = 0.0
    age_s: float = 0.0
    top: list[EmotionEpisode] = field(default_factory=list)

    def dominant(self, count: int = 2) -> list[tuple[str, float]]:
        return sorted(self.labels.items(), key=lambda item: item[1], reverse=True)[:count]

    def baseline(self) -> tuple[dict[str, float], float]:
        """Slow layer for the mixer: (labels, intensity) with a damped weight."""
        return dict(self.labels), min(1.0, self.intensity)

    def summary(self, max_chars: int = 220) -> str:
        """A compact first-person line for the prompt; empty when neutral."""
        if not self.top or not self.labels:
            return ""
        labels = " / ".join(
            f"{LABEL_CN.get(name, name)} {share:.2f}" for name, share in self.dominant(2)
        )
        pieces = [
            f"（参考，不是命令）现在的情绪：{labels}，强度 {self.intensity:.1f}，"
            f"从 {_human_age(self.age_s)}前开始。"
        ]
        # Only the tool self-report (or an episode with an explicit why) may
        # claim a reason; keyword sources are gone, but keep the guard.
        first = next(
            (ep for ep in self.top if ep.what and (ep.why or ep.source == "tool")),
            None,
        )
        if first is not None:
            reason = first.what
            if first.why:
                reason += f"——{first.why}"
            pieces.append(f"为什么：{reason}")
        second = next((ep for ep in self.top[1:] if ep.what), None)
        if second is not None:
            pieces.append(f"余波：{second.what}")
        return "".join(pieces)[:max_chars]


def _merge_labels(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    merged = dict(left)
    for label, share in right.items():
        merged[label] = max(merged.get(label, 0.0), share)
    return normalize_labels(merged)


def _human_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)} 秒"
    if seconds < 3600:
        return f"{int(seconds // 60)} 分钟"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} 小时"
    return f"{seconds / 86400:.1f} 天"
