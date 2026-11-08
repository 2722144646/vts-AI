"""Local chat transcript viewer.

A small always-on-top Tkinter window that tails ``data/chat/<user>.jsonl``.
Nothing here goes through the host agent: the file is read straight from disk.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except FileNotFoundError:
        pass
    return rows


class Viewer:
    def __init__(self, path: Path, limit: int = 80) -> None:
        import tkinter as tk

        self.tk = tk
        self.path = path
        self.limit = limit
        self.shown = -1
        self.root = tk.Tk()
        self.root.title("聊天记录（本地）")
        self.root.geometry("520x620")
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#181818")

        header = tk.Label(
            self.root,
            text=f"本地原文 · {path.name}",
            bg="#181818",
            fg="#8a8a8a",
            anchor="w",
            font=("Microsoft YaHei UI", 9),
        )
        header.pack(fill="x", padx=10, pady=(8, 0))

        frame = tk.Frame(self.root, bg="#181818")
        frame.pack(fill="both", expand=True, padx=10, pady=8)
        scroll = tk.Scrollbar(frame)
        scroll.pack(side="right", fill="y")
        self.text = tk.Text(
            frame,
            wrap="word",
            font=("Microsoft YaHei UI", 11),
            bg="#101010",
            fg="#e6e6e6",
            insertbackground="#e6e6e6",
            yscrollcommand=scroll.set,
            padx=8,
            pady=8,
        )
        self.text.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.text.yview)
        self.text.tag_configure("user", foreground="#9fb6d4")
        self.text.tag_configure("assistant", foreground="#f0b9c8")
        self.text.tag_configure("time", foreground="#666666", font=("Microsoft YaHei UI", 8))
        self.text.configure(state="disabled")

        self.poll()

    def _render(self, rows: list[dict[str, Any]]) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        for row in rows[-self.limit :]:
            stamp = str(row.get("at", ""))[11:19]
            user = str(row.get("user") or "")
            reply = str(row.get("assistant") or "")
            if stamp:
                self.text.insert("end", f"{stamp}\n", "time")
            if user:
                self.text.insert("end", "你：", "user")
                self.text.insert("end", user + "\n")
            if reply:
                self.text.insert("end", "她：", "assistant")
                self.text.insert("end", reply + "\n\n")
        self.text.configure(state="disabled")
        self.text.see("end")

    def poll(self) -> None:
        rows = load_rows(self.path)
        if len(rows) != self.shown:
            self.shown = len(rows)
            self._render(rows)
        self.root.after(1000, self.poll)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="path to the chat jsonl")
    args = parser.parse_args()
    Viewer(Path(args.file)).root.mainloop()


if __name__ == "__main__":
    main()
