"""Command line entry points."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

from .audio.engine import make_clock
from .bridge.engine import AvatarEngine
from .bridge.server import run_bridge
from .config import ensure_dirs, load_config
from .ndi.capture import NDICapture, list_sources
from .performance.recorder import record_expression
from .vts.client import VTSClient
from .vts.probe import run_probe
from .vts.profile import ModelProfile, profile_path

log = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def _cmd_probe(args) -> int:
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    client = VTSClient(cfg.vts)
    try:
        profile = await run_probe(cfg, client, request_token=args.request_token)
    except Exception as exc:
        log.error("probe failed: %s", exc)
        print(
            "\nVTS probe failed. Check:\n"
            "  1. VTube Studio is running.\n"
            "  2. Settings -> API -> Allow Plugin API access is ON.\n"
            "  3. The port matches config.toml (default 8001).\n"
            "  4. If this is the first run, approve the plugin popup in VTS.\n",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(profile.summary(), ensure_ascii=False, indent=2))
    return 0


async def _cmd_demo(args) -> int:
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.no_audio:
        cfg.audio.enabled = False
    engine = AvatarEngine(cfg, on_event=_print_event)
    await engine.start()
    try:
        await engine.begin_utterance("demo")
        await engine.feed_text(args.text)
        await engine.end_utterance()
        timeout = time.monotonic() + args.seconds
        while time.monotonic() < timeout:
            await asyncio.sleep(0.1)
    finally:
        await engine.close()
    return 0


async def _cmd_record(args) -> int:
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    client = VTSClient(cfg.vts)
    await client.connect()
    ndi = None
    session = None
    try:
        model = await client.current_model()
        model_id = str(model.get("modelID", ""))
        profile = ModelProfile.load(profile_path(cfg.profile_dir, model_id))
        if args.ndi:
            ndi = NDICapture(source_hint=cfg.ndi.source, snapshot_dir=cfg.snapshot_dir)
            if not ndi.open(wait_ms=4000):
                print("NDI source not found; continuing without frames.", file=sys.stderr)
                ndi = None
        if args.session_root:
            root = Path(args.session_root)
            if not root.is_absolute():
                root = cfg.root / root
        else:
            root = cfg.root / "data" / "demos"
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in args.name)
        session = root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe or 'expression'}"
        print(f"Session: {session}", flush=True)
        print("Read-only recording. Keep neutral until the voice cue, then do the expression.", flush=True)

        def speak(text: str) -> None:
            try:
                subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-Command",
                        "Add-Type -AssemblyName System.Speech; "
                        f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')",
                    ],
                    timeout=6,
                    check=False,
                )
            except Exception:
                pass

        def session_cue() -> None:
            try:
                import winsound

                winsound.Beep(700, 120)
            except Exception:
                pass
            print("SESSION", flush=True)

        def start_cue() -> None:
            try:
                import winsound

                winsound.Beep(1000, 120)
            except Exception:
                pass
            speak(f"开始，{args.name}")
            print("START", flush=True)

        def done_cue() -> None:
            try:
                import winsound

                winsound.Beep(660, 120)
            except Exception:
                pass
            speak("停")
            print("DONE", flush=True)

        summary = await record_expression(
            client,
            args.name,
            duration_s=args.seconds,
            on_start=start_cue,
            on_done=done_cue,
            on_session_start=session_cue,
            ndi=ndi,
            session_dir=session,
            audio_device=args.audio_device if args.audio else None,
        )
        profile.learned_expressions[args.name] = summary
        direct = {
            key: float(info["peak"])
            for key, info in summary.get("input_deltas", {}).items()
            if abs(float(info.get("delta", 0.0))) >= 0.02
        }
        profile.expression_map[args.name] = direct
        profile.save(profile_path(cfg.profile_dir, model_id))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        if ndi is not None:
            ndi.close()
        await client.close()


async def _print_event(payload: dict) -> None:
    if payload.get("type") in {"ready", "playback", "error", "cancelled"}:
        print(json.dumps(payload, ensure_ascii=False))


async def _cmd_serve(args) -> int:
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.no_audio:
        cfg.audio.enabled = False
    await run_bridge(cfg)
    return 0


def _cmd_ndi(args) -> int:
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.list:
        sources = list_sources(args.wait_ms)
        print(json.dumps(sources, ensure_ascii=False, indent=2))
        return 0
    capture = NDICapture(source_hint=args.source or cfg.ndi.source, snapshot_dir=cfg.snapshot_dir)
    try:
        if not capture.open(wait_ms=args.wait_ms):
            print("No NDI sources found. Enable NDI output in VTS first.", file=sys.stderr)
            return 2
        metrics = capture.snapshot()
        if metrics is None:
            print("NDI source found but no video frame arrived.", file=sys.stderr)
            return 3
        print(json.dumps(metrics.__dict__, ensure_ascii=False, indent=2))
        return 0
    finally:
        capture.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dsh-vts", description="DSH -> VTube Studio avatar bridge")
    parser.add_argument("--config", help="path to config.toml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="probe VTS parameters and write data/contract.json")
    probe.add_argument("--request-token", action="store_true", help="force a new VTS plugin token popup")

    serve = sub.add_parser("serve", help="run the local WebSocket bridge")
    serve.add_argument("--no-audio", action="store_true")

    demo = sub.add_parser("demo", help="run one line through TTS/VTS without DSH")
    demo.add_argument("text", nargs="?", default="你好，我是 dsh。现在，我正在用 VTube Studio 和你说话。")
    demo.add_argument("--seconds", type=float, default=6.0)
    demo.add_argument("--no-audio", action="store_true")

    record = sub.add_parser("record", help="read-only capture of a user-performed expression")
    record.add_argument("--name", required=True, help="learned expression name")
    record.add_argument("--seconds", type=float, default=4.0)
    record.add_argument("--ndi", action="store_true", help="also save synchronized NDI frames")
    record.add_argument("--audio", action="store_true", help="also save microphone audio")
    record.add_argument("--audio-device", type=int, default=None)
    record.add_argument("--session-root", default="", help="default: data/demos")

    ndi = sub.add_parser("ndi", help="list or capture the VTS NDI feed")
    ndi.add_argument("--list", action="store_true")
    ndi.add_argument("--source", default="")
    ndi.add_argument("--wait-ms", type=int, default=3000)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        if args.command == "probe":
            return asyncio.run(_cmd_probe(args))
        if args.command == "serve":
            return asyncio.run(_cmd_serve(args))
        if args.command == "demo":
            return asyncio.run(_cmd_demo(args))
        if args.command == "record":
            return asyncio.run(_cmd_record(args))
        if args.command == "ndi":
            return _cmd_ndi(args)
    except KeyboardInterrupt:
        return 130
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
