"""Small local settings panel for the LLM backend.

Edit ``[llm]`` in config.toml without hand-editing TOML, then restart the
bridge (the config is read once at startup). Runs entirely locally; the
window has no network access except the explicit "test" button.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path


def _section_bounds(text: str, section: str) -> tuple[int, int] | None:
    header = re.search(rf"^\[{re.escape(section)}\]\s*$", text, re.M)
    if not header:
        return None
    start = header.end()
    nxt = re.search(r"^\[[^\]]+\]\s*$", text[start:], re.M)
    end = start + nxt.start() if nxt else len(text)
    return start, end


def _format_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(str(value), ensure_ascii=False)


def set_section_value(text: str, section: str, key: str, value) -> str:
    """Set one key inside a TOML section, keeping everything else untouched."""
    bounds = _section_bounds(text, section)
    if bounds is None:
        text = text.rstrip() + f"\n\n[{section}]\n{key} = {_format_value(value)}\n"
        return text
    start, end = bounds
    body = text[start:end]
    pattern = re.compile(rf"^{re.escape(key)}\s*=.*$", re.M)
    if pattern.search(body):
        body = pattern.sub(f"{key} = {_format_value(value)}", body, count=1)
    else:
        body = body.rstrip("\n") + f"\n{key} = {_format_value(value)}\n"
    return text[:start] + body + text[end:]


def read_section(path: Path, section: str) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    bounds = _section_bounds(text, section)
    if bounds is None:
        return {}
    values: dict[str, str] = {}
    for line in text[bounds[0] : bounds[1]].splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw = line.split("=", 1)
        raw = raw.strip()
        if raw.startswith('"') and raw.endswith('"'):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = raw.strip('"')
        values[key.strip()] = raw
    return values


class Panel:
    def __init__(self, config_path: Path) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.path = config_path
        self.values = read_section(config_path, "llm")
        self.root = tk.Tk()
        self.root.title("dsh-vts-avatar · 模型设置")
        self.root.geometry("520x380")
        self.root.attributes("-topmost", True)

        body = ttk.Frame(self.root, padding=14)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)

        self.vars: dict[str, object] = {}
        rows = [
            ("backend", "后端", ["ollama", "openai", "none"]),
            ("endpoint", "接口地址", None),
            ("api_key", "API Key", None),
            ("model", "对话模型", None),
            ("director_model", "导演模型", None),
        ]
        for row, (key, label, choices) in enumerate(rows):
            ttk.Label(body, text=label).grid(row=row, column=0, sticky="w", pady=6)
            if choices:
                var = tk.StringVar(value=str(self.values.get(key, choices[0])))
                widget = ttk.Combobox(body, textvariable=var, values=choices, state="readonly")
            else:
                var = tk.StringVar(value=str(self.values.get(key, "")))
                widget = ttk.Entry(body, textvariable=var)
            widget.grid(row=row, column=1, sticky="ew", pady=6)
            self.vars[key] = var

        self.enabled = tk.BooleanVar(value=str(self.values.get("enabled", "false")).lower() == "true")
        self.director = tk.BooleanVar(
            value=str(self.values.get("director_enabled", "false")).lower() == "true"
        )
        ttk.Checkbutton(body, text="启用本地模型（私有对话）", variable=self.enabled).grid(
            row=len(rows), column=1, sticky="w", pady=4
        )
        ttk.Checkbutton(body, text="启用待机导演（常驻显存）", variable=self.director).grid(
            row=len(rows) + 1, column=1, sticky="w", pady=4
        )

        buttons = ttk.Frame(body)
        buttons.grid(row=len(rows) + 2, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Button(buttons, text="保存", command=self.save).pack(side="left")
        ttk.Button(buttons, text="测试连接", command=self.test).pack(side="left", padx=8)

        self.status = ttk.Label(body, text="改完保存，重启桥接生效。", foreground="#555")
        self.status.grid(row=len(rows) + 3, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def save(self) -> str:
        text = self.path.read_text(encoding="utf-8")
        updates = {k: str(v.get()) for k, v in self.vars.items()}
        updates["enabled"] = self.enabled.get()
        updates["director_enabled"] = self.director.get()
        for key, value in updates.items():
            text = set_section_value(text, "llm", key, value)
        self.path.write_text(text, encoding="utf-8")
        self.status.configure(text="已保存。重启桥接后生效。", foreground="#1a7f37")
        return str(self.path)

    def test(self) -> bool:
        backend = str(self.vars["backend"].get())
        endpoint = str(self.vars["endpoint"].get()).rstrip("/")
        key = str(self.vars["api_key"].get())
        try:
            if backend == "ollama":
                with urllib.request.urlopen(endpoint + "/api/tags", timeout=3) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                names = [m.get("name", "") for m in data.get("models", [])]
                self.status.configure(
                    text=f"Ollama 可用，已装 {len(names)} 个模型。", foreground="#1a7f37"
                )
                return True
            if backend == "openai":
                base = endpoint + ("" if endpoint.endswith("/v1") else "/v1")
                headers = {"Authorization": f"Bearer {key}"} if key else {}
                request = urllib.request.Request(base + "/models", headers=headers)
                with urllib.request.urlopen(request, timeout=5) as resp:
                    json.loads(resp.read().decode("utf-8"))
                self.status.configure(text="接口可访问。", foreground="#1a7f37")
                return True
            self.status.configure(text="none：模型已关闭。", foreground="#555")
            return True
        except Exception as exc:  # noqa: BLE001 - show it to the user
            self.status.configure(text=f"连接失败：{exc}", foreground="#b42318")
            return False

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()
    Panel(Path(args.config)).run()


if __name__ == "__main__":
    main()
