"""Model capability profile derived from a VTS contract probe."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from ..config import AvatarConfig


@dataclass(slots=True)
class ParameterInfo:
    name: str
    added_by: str = ""
    value: float = 0.0
    minimum: float = 0.0
    maximum: float = 1.0
    default: float = 0.0

    @classmethod
    def from_vts(cls, raw: dict[str, Any]) -> "ParameterInfo":
        return cls(
            name=str(raw.get("name", "")),
            added_by=str(raw.get("addedBy", "")),
            value=float(raw.get("value", 0.0)),
            minimum=float(raw.get("min", 0.0)),
            maximum=float(raw.get("max", 1.0)),
            default=float(raw.get("defaultValue", 0.0)),
        )


@dataclass
class ModelProfile:
    model_name: str = ""
    model_id: str = ""
    input_params: dict[str, ParameterInfo] = field(default_factory=dict)
    live2d_params: dict[str, ParameterInfo] = field(default_factory=dict)
    probes: dict[str, dict[str, Any]] = field(default_factory=dict)
    range_tests: dict[str, dict[str, Any]] = field(default_factory=dict)
    expression_map: dict[str, dict[str, float]] = field(default_factory=dict)
    learned_expressions: dict[str, dict[str, Any]] = field(default_factory=dict)
    mouth_backend: str = "auto"
    # Per-model tuning survives model switches. Keys match AvatarConfig fields,
    # e.g. head_gain, mouth_gain, jaw_gain, expression_gain.
    tuning: dict[str, float] = field(default_factory=dict)
    # Live2D params that visibly moved when a tracking input was probed.
    driven_live2d: dict[str, list[str]] = field(default_factory=dict)

    @property
    def input_names(self) -> set[str]:
        return set(self.input_params)

    @property
    def live2d_names(self) -> set[str]:
        return set(self.live2d_params)

    def has(self, name: str) -> bool:
        return name in self.input_params

    def clamp(self, name: str, value: float) -> float:
        info = self.input_params.get(name)
        if info is None:
            return float(value)
        return max(info.minimum, min(info.maximum, float(value)))

    def filter(self, params: dict[str, float]) -> dict[str, float]:
        """Drop parameters that the current model does not expose.

        Probe results are authoritative. If no contract exists, keep values so
        the first run can surface VTS errors instead of silently doing nothing.
        """
        if not self.input_params:
            return {k: float(v) for k, v in params.items() if v == v}
        out: dict[str, float] = {}
        for key, value in params.items():
            if key in self.input_params:
                out[key] = self.clamp(key, value)
        return out

    def choose_mouth_backend(self, avatar: AvatarConfig, requested: str = "auto") -> str:
        if requested != "auto":
            return requested
        voices = [avatar.voice_a, avatar.voice_i, avatar.voice_u, avatar.voice_e, avatar.voice_o]
        if all(name in self.input_params for name in voices):
            return "vowel"
        if avatar.mouth_open in self.input_params:
            return "continuous"
        return "none"

    def voice_params(self, avatar: AvatarConfig) -> list[str]:
        return [avatar.voice_a, avatar.voice_i, avatar.voice_u, avatar.voice_e, avatar.voice_o]

    def summary(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "input_count": len(self.input_params),
            "live2d_count": len(self.live2d_params),
            "mouth_backend": self.mouth_backend,
            "tuning": dict(self.tuning),
            "has_voice": all(
                name in self.input_params
                for name in ("VoiceA", "VoiceI", "VoiceU", "VoiceE", "VoiceO")
            ),
            "has_body": all(name in self.input_params for name in ("DSHBodyX", "DSHBodyY", "DSHBodyZ")),
            "probes": {
                name: {
                    "status": p.get("status"),
                    "error_id": p.get("error_id"),
                    "moved": p.get("moved", [])[:8],
                }
                for name, p in self.probes.items()
            },
        }

    @classmethod
    def load(cls, path: Path) -> "ModelProfile":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        profile = cls(
            model_name=str(raw.get("model", {}).get("modelName", "")),
            model_id=str(raw.get("model", {}).get("modelID", "")),
            mouth_backend=str(raw.get("mouth_backend", "auto")),
            tuning={str(k): float(v) for k, v in (raw.get("tuning") or {}).items()},
            range_tests=dict(raw.get("range_tests") or {}),
            expression_map={
                str(preset): {str(name): float(value) for name, value in values.items()}
                for preset, values in (raw.get("expression_map") or {}).items()
            },
            learned_expressions=dict(raw.get("learned_expressions") or {}),
        )
        for item in raw.get("input_parameters", []):
            info = ParameterInfo.from_vts(item)
            if info.name:
                profile.input_params[info.name] = info
        for item in raw.get("live2d_parameters", []):
            info = ParameterInfo.from_vts(item)
            if info.name:
                profile.live2d_params[info.name] = info
        profile.probes = raw.get("probes", {})
        profile.driven_live2d = raw.get("driven_live2d", {})
        return profile

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "model": {"modelName": self.model_name, "modelID": self.model_id},
                    "mouth_backend": self.mouth_backend,
                    "tuning": dict(self.tuning),
                    "input_parameters": [
                        {
                            "name": p.name,
                            "addedBy": p.added_by,
                            "value": p.value,
                            "min": p.minimum,
                            "max": p.maximum,
                            "defaultValue": p.default,
                        }
                        for p in self.input_params.values()
                    ],
                    "live2d_parameters": [
                        {
                            "name": p.name,
                            "value": p.value,
                            "min": p.minimum,
                            "max": p.maximum,
                            "defaultValue": p.default,
                        }
                        for p in self.live2d_params.values()
                    ],
                    "probes": self.probes,
                    "range_tests": self.range_tests,
                    "expression_map": self.expression_map,
                    "learned_expressions": self.learned_expressions,
                    "driven_live2d": self.driven_live2d,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def profile_path(directory: Path, model_id: str) -> Path:
    """Stable per-model filename; never overwrite another model's tuning."""
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in model_id)
    return directory / f"{safe or 'unknown-model'}.json"


def load_for_model(directory: Path, model_id: str) -> ModelProfile:
    path = profile_path(directory, model_id)
    if path.exists():
        return ModelProfile.load(path)
    return ModelProfile(model_id=model_id)
