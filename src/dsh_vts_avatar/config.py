"""Typed configuration for the dsh VTS avatar bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class VTSConfig:
    host: str = "127.0.0.1"
    port: int = 8001
    plugin_name: str = "dsh VTS Avatar"
    plugin_developer: str = "dsh"
    token_path: Path = Path("data/vts_token.json")
    contract_path: Path = Path("data/contract.json")
    profile_dir: Path = Path("data/profiles")
    allow_add_fallback: bool = False

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"


@dataclass(slots=True)
class TTSConfig:
    voice_model: Path = Path("models/zh_CN-huayan-medium.onnx")
    voice_config: Path | None = Path("models/zh_CN-huayan-medium.onnx.json")
    language: str = "zh"
    length_scale: float = 1.0
    sentence_silence: float = 0.08
    postprocess: bool = False
    ffmpeg_path: Path = Path("D:/ruanjian/oopz/ffmpeg.exe")
    distill_model: Path | None = None
    distill_voice_config: Path | None = None
    distill_length_scale: float = 0.61
    noise_scale: float | None = None
    noise_w_scale: float | None = None


@dataclass(slots=True)
class AudioConfig:
    device: int | None = None
    sample_rate: int = 48000
    buffer_ms: int = 140
    av_offset_ms: int = 0
    auto_calibrate: bool = True
    calibration_volume: float = 0.18
    sync_correction_ms: float = 0.0
    enabled: bool = True


@dataclass(slots=True)
class AvatarConfig:
    persona_name: str = "dsh"
    expressiveness: float = 0.65
    head_gain: float = 1.0
    head_motion: str = "minjerk"
    head_max_speed: float = 35.0
    head_spring_hz: float = 1.9
    head_spring_zeta: float = 1.0
    mouth_gain: float = 1.0
    gesture_rate: float = 0.4
    # Idle liveliness: 0 = quiet, 1 = lively. User answer 2026-11-06:
    # 有存在感 / 更活泼 一半一半.
    idle_energy: float = 0.55
    # Expression strength: natural 70% / exaggerated 30% (user answer).
    exaggeration_share: float = 0.30
    exaggeration_gain: float = 1.30
    # L2 mood baseline under the transient affect; 0 = off (old behavior).
    baseline_weight: float = 0.0
    blink_per_min: float = 14.0
    mouth_backend: str = "auto"
    mouth_open: str = "MouthOpen"
    mouth_smile: str = "MouthSmile"
    jaw_open: str = "JawOpen"
    mouth_funnel: str = "MouthFunnel"
    mouth_pucker: str = "MouthPucker"
    mouth_press: str = "MouthPressLipOpen"
    mouth_shrug: str = "MouthShrug"
    mouth_x: str = "MouthX"
    voice_silence: str = "VoiceSilence"
    brow_inner: str = "BrowInnerUp"
    eye_left: str = "EyeOpenLeft"
    eye_right: str = "EyeOpenRight"
    head_x: str = "FaceAngleX"
    head_y: str = "FaceAngleY"
    head_z: str = "FaceAngleZ"
    body_x: str = "DSHBodyX"
    body_y: str = "DSHBodyY"
    body_z: str = "DSHBodyZ"
    voice_a: str = "VoiceA"
    voice_i: str = "VoiceI"
    voice_u: str = "VoiceU"
    voice_e: str = "VoiceE"
    voice_o: str = "VoiceO"

    @property
    def voice_params(self) -> dict[str, str]:
        return {
            "A": self.voice_a,
            "I": self.voice_i,
            "U": self.voice_u,
            "E": self.voice_e,
            "O": self.voice_o,
        }


@dataclass(slots=True)
class NDIConfig:
    source: str = ""
    snapshot_dir: Path = Path("data/ndi")
    snapshot_interval_ms: int = 500


@dataclass(slots=True)
class BridgeConfig:
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass(slots=True)
class RVCConfig:
    enabled: bool = False
    base_url: str = "http://127.0.0.1:9880"
    api_key_file: Path = Path("E:/feiyu/service/.runtime/voice-service/api_key.txt")
    voice: str = "feiyu"
    cache_dir: Path = Path("data/rvc_cache")
    timeout_s: float = 300.0


@dataclass(slots=True)
class EmotionConfig:
    """Emotion pipeline selector.

    ``legacy`` keeps presets + priority + keyword fallback (v0.5.0 behavior).
    ``affect`` switches to the multi-source evidence pipeline. Phase 0 only
    adds the switch; the affect path arrives in later phases.
    """

    mode: str = "legacy"

    VALID_MODES = ("legacy", "affect")

    @property
    def is_affect(self) -> bool:
        return self.mode == "affect"


@dataclass(slots=True)
class LLMConfig:
    """Local Ollama model used for the private (uncensored) speech path."""

    enabled: bool = False
    backend: str = "ollama"
    endpoint: str = "http://127.0.0.1:11434"
    api_key: str = ""
    model: str = "qwen3-8b-abliterated"
    timeout_s: float = 180.0
    history_turns: int = 8
    persona: str = ""
    director_enabled: bool = False
    director_model: str = "qwen3:8b"
    director_timeout_s: float = 8.0


@dataclass(slots=True)
class Config:
    root: Path
    vts: VTSConfig = field(default_factory=VTSConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    avatar: AvatarConfig = field(default_factory=AvatarConfig)
    rvc: RVCConfig = field(default_factory=RVCConfig)
    ndi: NDIConfig = field(default_factory=NDIConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def resolve(self, path: Path | None) -> Path | None:
        if path is None:
            return None
        return path if path.is_absolute() else (self.root / path)

    @property
    def token_path(self) -> Path:
        return self.resolve(self.vts.token_path)  # type: ignore[return-value]

    @property
    def contract_path(self) -> Path:
        return self.resolve(self.vts.contract_path)  # type: ignore[return-value]

    @property
    def profile_dir(self) -> Path:
        return self.resolve(self.vts.profile_dir)  # type: ignore[return-value]

    @property
    def voice_model(self) -> Path:
        return self.resolve(self.tts.voice_model)  # type: ignore[return-value]

    @property
    def voice_config(self) -> Path | None:
        return self.resolve(self.tts.voice_config)

    @property
    def snapshot_dir(self) -> Path:
        return self.resolve(self.ndi.snapshot_dir)  # type: ignore[return-value]


def _section(data: dict, name: str) -> dict:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def _parse_audio_device(value: object) -> int | None:
    """Accept an explicit device index, or None/"default"/"auto" for the system default."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "default", "auto", "system"}:
        return None
    return int(value)


def load_config(path: str | Path | None = None) -> Config:
    """Load TOML config; relative paths resolve against the config file directory."""
    config_path = Path(path) if path else ROOT / "config.toml"
    config_path = config_path.expanduser().resolve()
    root = config_path.parent
    raw: dict = {}
    if config_path.exists():
        with config_path.open("rb") as fh:
            raw = tomllib.load(fh)

    vts_raw = _section(raw, "vts")
    tts_raw = _section(raw, "tts")
    audio_raw = _section(raw, "audio")
    avatar_raw = _section(raw, "avatar")
    ndi_raw = _section(raw, "ndi")
    bridge_raw = _section(raw, "bridge")
    rvc_raw = _section(raw, "rvc")
    emotion_raw = _section(raw, "emotion")
    llm_raw = _section(raw, "llm")

    cfg = Config(root=root)
    cfg.vts = VTSConfig(
        host=str(vts_raw.get("host", "127.0.0.1")),
        port=int(vts_raw.get("port", 8001)),
        plugin_name=str(vts_raw.get("plugin_name", "dsh VTS Avatar")),
        plugin_developer=str(vts_raw.get("plugin_developer", "dsh")),
        token_path=Path(vts_raw.get("token_path", "data/vts_token.json")),
        contract_path=Path(vts_raw.get("contract_path", "data/contract.json")),
        profile_dir=Path(vts_raw.get("profile_dir", "data/profiles")),
        allow_add_fallback=bool(vts_raw.get("allow_add_fallback", False)),
    )
    cfg.tts = TTSConfig(
        voice_model=Path(tts_raw.get("voice_model", "models/zh_CN-huayan-medium.onnx")),
        voice_config=Path(tts_raw["voice_config"]) if tts_raw.get("voice_config") else None,
        language=str(tts_raw.get("language", "zh")),
        length_scale=float(tts_raw.get("length_scale", 1.0)),
        sentence_silence=float(tts_raw.get("sentence_silence", 0.08)),
        postprocess=bool(tts_raw.get("postprocess", False)),
        ffmpeg_path=Path(tts_raw.get("ffmpeg_path", "D:/ruanjian/oopz/ffmpeg.exe")),
        distill_model=Path(tts_raw["distill_model"]) if tts_raw.get("distill_model") else None,
        distill_voice_config=Path(tts_raw["distill_voice_config"]) if tts_raw.get("distill_voice_config") else None,
        distill_length_scale=float(tts_raw.get("distill_length_scale", 0.61)),
        noise_scale=float(tts_raw["noise_scale"]) if tts_raw.get("noise_scale") is not None else None,
        noise_w_scale=float(tts_raw["noise_w_scale"]) if tts_raw.get("noise_w_scale") is not None else None,
    )
    cfg.audio = AudioConfig(
        device=_parse_audio_device(audio_raw.get("device")),
        sample_rate=int(audio_raw.get("sample_rate", 48000)),
        buffer_ms=int(audio_raw.get("buffer_ms", 140)),
        av_offset_ms=int(audio_raw.get("av_offset_ms", 0)),
        auto_calibrate=bool(audio_raw.get("auto_calibrate", True)),
        calibration_volume=float(audio_raw.get("calibration_volume", 0.18)),
        sync_correction_ms=float(audio_raw.get("sync_correction_ms", 0.0)),
        enabled=bool(audio_raw.get("enabled", True)),
    )
    cfg.avatar = AvatarConfig(
        persona_name=str(avatar_raw.get("persona_name", "dsh")),
        expressiveness=float(avatar_raw.get("expressiveness", 0.65)),
        head_gain=float(avatar_raw.get("head_gain", 1.0)),
        head_motion=str(avatar_raw.get("head_motion", "minjerk")),
        head_max_speed=float(avatar_raw.get("head_max_speed", 35.0)),
        head_spring_hz=float(avatar_raw.get("head_spring_hz", 1.9)),
        head_spring_zeta=float(avatar_raw.get("head_spring_zeta", 1.0)),
        mouth_gain=float(avatar_raw.get("mouth_gain", 1.0)),
        gesture_rate=float(avatar_raw.get("gesture_rate", 0.4)),
        idle_energy=float(avatar_raw.get("idle_energy", 0.55)),
        exaggeration_share=float(avatar_raw.get("exaggeration_share", 0.30)),
        exaggeration_gain=float(avatar_raw.get("exaggeration_gain", 1.30)),
        baseline_weight=float(avatar_raw.get("baseline_weight", 0.0)),
        blink_per_min=float(avatar_raw.get("blink_per_min", 14.0)),
        mouth_backend=str(avatar_raw.get("mouth_backend", "auto")),
        mouth_open=str(avatar_raw.get("mouth_open", "MouthOpen")),
        mouth_smile=str(avatar_raw.get("mouth_smile", "MouthSmile")),
        jaw_open=str(avatar_raw.get("jaw_open", "JawOpen")),
        mouth_funnel=str(avatar_raw.get("mouth_funnel", "MouthFunnel")),
        mouth_pucker=str(avatar_raw.get("mouth_pucker", "MouthPucker")),
        mouth_press=str(avatar_raw.get("mouth_press", "MouthPressLipOpen")),
        mouth_shrug=str(avatar_raw.get("mouth_shrug", "MouthShrug")),
        mouth_x=str(avatar_raw.get("mouth_x", "MouthX")),
        voice_silence=str(avatar_raw.get("voice_silence", "VoiceSilence")),
        brow_inner=str(avatar_raw.get("brow_inner", "BrowInnerUp")),
        eye_left=str(avatar_raw.get("eye_left", "EyeOpenLeft")),
        eye_right=str(avatar_raw.get("eye_right", "EyeOpenRight")),
        head_x=str(avatar_raw.get("head_x", "FaceAngleX")),
        head_y=str(avatar_raw.get("head_y", "FaceAngleY")),
        head_z=str(avatar_raw.get("head_z", "FaceAngleZ")),
        body_x=str(avatar_raw.get("body_x", "DSHBodyX")),
        body_y=str(avatar_raw.get("body_y", "DSHBodyY")),
        body_z=str(avatar_raw.get("body_z", "DSHBodyZ")),
        voice_a=str(avatar_raw.get("voice_a", "VoiceA")),
        voice_i=str(avatar_raw.get("voice_i", "VoiceI")),
        voice_u=str(avatar_raw.get("voice_u", "VoiceU")),
        voice_e=str(avatar_raw.get("voice_e", "VoiceE")),
        voice_o=str(avatar_raw.get("voice_o", "VoiceO")),
    )
    cfg.rvc = RVCConfig(
        enabled=bool(rvc_raw.get("enabled", False)),
        base_url=str(rvc_raw.get("base_url", "http://127.0.0.1:9880")),
        api_key_file=Path(
            rvc_raw.get(
                "api_key_file",
                "E:/feiyu/service/.runtime/voice-service/api_key.txt",
            )
        ),
        voice=str(rvc_raw.get("voice", "feiyu")),
        cache_dir=Path(rvc_raw.get("cache_dir", "data/rvc_cache")),
        timeout_s=float(rvc_raw.get("timeout_s", 300.0)),
    )
    cfg.ndi = NDIConfig(
        source=str(ndi_raw.get("source", "")),
        snapshot_dir=Path(ndi_raw.get("snapshot_dir", "data/ndi")),
        snapshot_interval_ms=int(ndi_raw.get("snapshot_interval_ms", 500)),
    )
    cfg.bridge = BridgeConfig(
        host=str(bridge_raw.get("host", "127.0.0.1")),
        port=int(bridge_raw.get("port", 8765)),
    )
    mode = str(emotion_raw.get("mode", "legacy")).strip().lower()
    if mode not in EmotionConfig.VALID_MODES:
        mode = "legacy"
    cfg.emotion = EmotionConfig(mode=mode)
    cfg.llm = LLMConfig(
        enabled=bool(llm_raw.get("enabled", False)),
        backend=str(llm_raw.get("backend", "ollama")),
        endpoint=str(llm_raw.get("endpoint", "http://127.0.0.1:11434")),
        api_key=str(llm_raw.get("api_key", "")),
        model=str(llm_raw.get("model", "qwen3-8b-abliterated")),
        timeout_s=float(llm_raw.get("timeout_s", 180.0)),
        history_turns=int(llm_raw.get("history_turns", 8)),
        persona=str(llm_raw.get("persona", "")),
        director_enabled=bool(llm_raw.get("director_enabled", False)),
        director_model=str(llm_raw.get("director_model", "qwen3:8b")),
        director_timeout_s=float(llm_raw.get("director_timeout_s", 8.0)),
    )
    # Resolve every filesystem path once, against the config directory. This
    # keeps the bridge independent of the process current working directory.
    cfg.vts.token_path = cfg.resolve(cfg.vts.token_path)  # type: ignore[assignment]
    cfg.vts.contract_path = cfg.resolve(cfg.vts.contract_path)  # type: ignore[assignment]
    cfg.vts.profile_dir = cfg.resolve(cfg.vts.profile_dir)  # type: ignore[assignment]
    cfg.tts.voice_model = cfg.resolve(cfg.tts.voice_model)  # type: ignore[assignment]
    cfg.tts.voice_config = cfg.resolve(cfg.tts.voice_config)
    cfg.tts.ffmpeg_path = cfg.resolve(cfg.tts.ffmpeg_path)  # type: ignore[assignment]
    if cfg.tts.distill_model is not None:
        cfg.tts.distill_model = cfg.resolve(cfg.tts.distill_model)  # type: ignore[assignment]
    if cfg.tts.distill_voice_config is not None:
        cfg.tts.distill_voice_config = cfg.resolve(cfg.tts.distill_voice_config)  # type: ignore[assignment]
    cfg.rvc.api_key_file = cfg.resolve(cfg.rvc.api_key_file)  # type: ignore[assignment]
    cfg.rvc.cache_dir = cfg.resolve(cfg.rvc.cache_dir)  # type: ignore[assignment]
    cfg.ndi.snapshot_dir = cfg.resolve(cfg.ndi.snapshot_dir)  # type: ignore[assignment]
    return cfg


def ensure_dirs(cfg: Config) -> None:
    for path in (
        cfg.token_path.parent,
        cfg.contract_path.parent,
        cfg.profile_dir,
        cfg.snapshot_dir,
        cfg.rvc.cache_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
