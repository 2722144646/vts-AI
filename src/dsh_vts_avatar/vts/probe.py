"""VTS contract probe: what can we actually control on this model?"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from ..config import Config
from .client import VTSClient, VTSError
from .profile import ModelProfile, ParameterInfo, profile_path

log = logging.getLogger(__name__)

CANDIDATE_RE = re.compile(
    r"(?i)(mouth|jaw|voice|faceangle|faceposition|eye|brow|cheek|angry|smile|tongue|"
    r"^vb|vbridger|dsh)"
)


def _interesting(name: str) -> bool:
    return bool(CANDIDATE_RE.search(name))


def _probe_value(info: ParameterInfo) -> float:
    span = info.maximum - info.minimum
    if span <= 0:
        return info.default
    target = info.default + 0.70 * (info.maximum - info.default)
    # For symmetric params such as FaceAngleX, this lands at a visible +21.
    return max(info.minimum, min(info.maximum, target))


def _diff(before: dict[str, float], after: dict[str, float], limit: int = 8) -> list[tuple[str, float]]:
    changed: list[tuple[str, float]] = []
    for name, value in after.items():
        old = before.get(name)
        if old is None:
            continue
        delta = value - old
        if abs(delta) >= 0.01:
            changed.append((name, delta))
    changed.sort(key=lambda item: abs(item[1]), reverse=True)
    return changed[:limit]


async def run_probe(
    cfg: Config,
    client: VTSClient,
    *,
    request_token: bool = False,
    sleep_s: float = 0.30,
) -> ModelProfile:
    await client.connect(request_token=request_token)
    try:
        model = await client.current_model()
        inputs = await client.list_input_params()
        live2d = await client.list_live2d_params()

        profile = ModelProfile(
            model_name=str(model.get("modelName", "")),
            model_id=str(model.get("modelID", "")),
        )
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

        profile.mouth_backend = profile.choose_mouth_backend(cfg.avatar, cfg.avatar.mouth_backend)
        log.info(
            "model=%s inputs=%d live2d=%d backend=%s",
            profile.model_name,
            len(profile.input_params),
            len(profile.live2d_params),
            profile.mouth_backend,
        )

        names = sorted(name for name in profile.input_params if _interesting(name))
        if not names:
            log.warning("no interesting input parameters found")
            profile.probes["_none"] = {"status": "no_candidates"}
            return profile

        before = {name: info.value for name, info in profile.live2d_params.items()}
        for name in names:
            info = profile.input_params[name]
            target = _probe_value(info)
            result: dict[str, Any] = {
                "target": target,
                "min": info.minimum,
                "max": info.maximum,
                "added_by": info.added_by,
            }
            try:
                await client.inject({name: target})
                await asyncio.sleep(sleep_s)
                after_raw = await client.list_live2d_params()
                after = {
                    str(item.get("name")): float(item.get("value", 0.0))
                    for item in after_raw.get("parameters", [])
                    if item.get("name")
                }
                moved = _diff(before, after)
                result["status"] = "ok" if moved else "no_visible_change"
                result["moved"] = [f"{param} {delta:+.3f}" for param, delta in moved]
                if moved:
                    profile.driven_live2d[name] = [param for param, _ in moved]
                before = after
            except VTSError as exc:
                result["status"] = "error"
                result["error_id"] = exc.error_id
                result["error"] = exc.vts_message
            except Exception as exc:
                result["status"] = "error"
                result["error"] = str(exc)
            finally:
                try:
                    await client.inject({name: info.default})
                    await asyncio.sleep(0.08)
                except Exception:
                    pass
            profile.probes[name] = result
            log.info(
                "probe %-22s %-16s %s",
                name,
                result.get("status"),
                result.get("moved", result.get("error", ""))[:3],
            )

        profile.mouth_backend = profile.choose_mouth_backend(cfg.avatar, cfg.avatar.mouth_backend)
        target = profile_path(cfg.profile_dir, profile.model_id)
        previous = ModelProfile.load(target)
        if previous.tuning:
            profile.tuning = dict(previous.tuning)
        profile.save(target)
        log.info("saved per-model profile to %s", target)
        return profile
    finally:
        await client.close()
