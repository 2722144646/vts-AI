"""Phase 4: weighted, non-prioritized evidence fusion.

The point of this module is the user's core requirement:

    freedom, truthfulness, coordinated five features, and no forced
    expression just because one source outranks another.

So every source is a *vote*, not a command. Votes are weighted by source
reliability, the source's own confidence, and freshness; the result is one
continuous state. Conflicting votes are allowed to coexist as a mixed state —
"don't know yet" is a legal answer.

The AI's own self-report (``tool``) carries the highest weight by design: it is
the only source that has actually read the conversation. A manual user
instruction is just another vote at half weight (``manual`` 0.5) — the user's
requirement is freedom, so no source may forcibly override another, not even an
explicit command.

Nothing here writes to VTS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from .schema import AffectEvidence, AffectState, derive_valence_arousal, normalize_labels

#: Base reliability per source. NDI is a correction channel, not an opinion.
#: ``tool`` (the AI's own self-report) is deliberately the highest: it is the
#: only vote backed by the actual conversation. ``manual`` is a user command,
#: but it is still only a mid-weight vote — nothing hard-overrides.
SOURCE_WEIGHTS: dict[str, float] = {
    "tool": 1.0,
    # Disabled producers (2026-11-07): tags and keywords no longer create
    # evidence, and the semantic classifier was cancelled. The weights stay
    # documented so old logs remain interpretable and a future producer can
    # opt back in without re-deciding the numbers.
    "tag": 0.85,
    "context": 0.75,
    "classifier": 0.7,
    "manual": 0.5,
    "keyword": 0.45,
    "ndi": 0.0,
}

DEFAULT_SOURCE_WEIGHT = 0.5


@dataclass(slots=True)
class FusionSettings:
    source_weights: dict[str, float] = field(default_factory=lambda: dict(SOURCE_WEIGHTS))
    #: Evidence older than this stops voting (mixer decay starts earlier).
    evidence_ttl_s: float = 10.0
    #: Optional stability guard: below this label distance, keep the previous
    #: labels instead of flip-flopping between two near-identical readings.
    switch_threshold: float = 0.15


class AffectFusion:
    """Accumulates evidence and derives one state; never a priority ladder."""

    def __init__(self, settings: FusionSettings | None = None, clock=time.time) -> None:
        self.settings = settings or FusionSettings()
        self.clock = clock
        self._evidence: list[AffectEvidence] = []
        self._last: AffectState | None = None

    # -- inputs ------------------------------------------------------------

    def add(self, evidence: AffectEvidence) -> AffectState:
        now = self.clock()
        self._prune(now)
        self._evidence.append(evidence)
        return self.state(now)

    def clear(self) -> None:
        self._evidence.clear()
        self._last = None

    @property
    def evidence_count(self) -> int:
        return len(self._evidence)

    # -- fusion ------------------------------------------------------------

    def state(self, now: float | None = None) -> AffectState:
        now = self.clock() if now is None else now
        self._prune(now)
        if not self._evidence:
            return AffectState.neutral()

        blended: dict[str, float] = {}
        total_weight = 0.0
        weighted_intensity = 0.0
        weighted_confidence = 0.0
        used = 0
        for evidence in self._evidence:
            weight = self._vote_weight(evidence, now)
            if weight <= 0.0:
                continue
            used += 1
            total_weight += weight
            weighted_intensity += weight * evidence.intensity
            weighted_confidence += weight * evidence.confidence
            for label, share in evidence.labels.items():
                blended[label] = blended.get(label, 0.0) + weight * share

        if total_weight <= 0.0 or used == 0:
            return AffectState.neutral()

        labels = normalize_labels(blended)
        labels = self._stabilize(labels)
        valence, arousal = derive_valence_arousal(labels)
        state = AffectState(
            labels=labels,
            intensity=_clamp(weighted_intensity / total_weight, 0.0, 1.0),
            confidence=_clamp(weighted_confidence / total_weight, 0.0, 1.0),
            valence=valence,
            arousal=arousal,
            source="fusion",
        )
        self._last = state
        return state

    # -- internals ---------------------------------------------------------

    def _prune(self, now: float) -> None:
        ttl = self.settings.evidence_ttl_s
        self._evidence = [e for e in self._evidence if now - e.timestamp <= ttl]

    def _vote_weight(self, evidence: AffectEvidence, now: float) -> float:
        reliability = self.settings.source_weights.get(evidence.source, DEFAULT_SOURCE_WEIGHT)
        if reliability <= 0.0:
            return 0.0
        age = max(0.0, now - evidence.timestamp)
        freshness = max(0.0, 1.0 - age / max(1e-6, self.settings.evidence_ttl_s))
        return evidence.confidence * reliability * freshness

    def _stabilize(self, labels: dict[str, float]) -> dict[str, float]:
        """Do not flip between two nearly identical readings every frame."""
        previous = self._last
        if previous is None:
            return labels
        distance = sum(
            abs(labels.get(key, 0.0) - previous.labels.get(key, 0.0))
            for key in set(labels) | set(previous.labels)
        ) / 2.0
        if distance < self.settings.switch_threshold:
            return dict(previous.labels)
        return labels


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
