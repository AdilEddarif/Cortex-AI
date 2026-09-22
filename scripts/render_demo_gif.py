"""Render the scripted demo as a terminal-style animated GIF (docs/demo.gif).

    python scripts/render_demo_gif.py [--out docs/demo.gif]

It runs ``organism.demo`` (deterministic), then draws its lines appearing one by one in a dark
terminal window: scene titles, what the person says, what CortexAI says, and the internal
events (dim) that produced each reply.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402  (only for its bundled monospace font)
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from organism.demo import Tour  # noqa: E402

W, H, PAD, SIZE, LINE_H = 1000, 560, 22, 15, 21
BG, BAR = (26, 26, 25), (44, 44, 42)
COLORS = {"scene": (109, 167, 236), "you": (245, 245, 240), "ai": (27, 175, 122), "inner": (140, 139, 132),
          "title": (195, 194, 183)}
PREFIX = {"scene": "", "you": "You: ", "ai": "CortexAI: ", "inner": "  · "}
HOLD_MS = {"scene": 1500, "you": 900, "ai": 1700, "inner": 1100}


def font(bold: bool = False) -> ImageFont.FreeTypeFont:
    base = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    return ImageFont.truetype(str(base / ("DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf")), SIZE)


def wrap(kind: str, text: str, cols: int) -> list[tuple[str, str]]:
    first = PREFIX[kind] + text
    indent = " " * (len(PREFIX[kind]) if kind != "scene" else 0)
    rows = textwrap.wrap(first, cols, subsequent_indent=indent) or [""]
    return [(kind, r) for r in rows]


def render(lines: list[tuple[str, str]], out: Path) -> None:
    regular, bold = font(), font(bold=True)
    cols = (W - 2 * PAD) // int(regular.getlength("M"))
    visible = (H - 60 - PAD) // LINE_H
    shown: list[tuple[str, str]] = []
    frames, durations = [], []

    def frame() -> Image.Image:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, 34], fill=BAR)
        for i, c in enumerate([(236, 104, 94), (245, 191, 79), (98, 197, 84)]):
            d.ellipse([14 + i * 22, 11, 26 + i * 22, 23], fill=c)
        d.text((W // 2, 17), "python -m organism demo", fill=COLORS["title"], font=regular, anchor="mm")
        y = 48
        for kind, row in shown[-visible:]:
            d.text((PAD, y), row, fill=COLORS[kind], font=bold if kind in ("scene", "ai") else regular)
            y += LINE_H
        return img

    frames.append(frame())
    durations.append(900)
    for kind, text in lines:
        if kind == "scene" and shown:
            shown.append(("inner", ""))
        shown.extend(wrap(kind, text, cols))
        frames.append(frame())
        durations.append(HOLD_MS[kind])
    durations[-1] = 4000
    out.parent.mkdir(parents=True, exist_ok=True)
    pal = [f.convert("P", palette=Image.Palette.ADAPTIVE, colors=32) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/demo.gif")
    args = ap.parse_args()
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    import asyncio
    tour = Tour(pace=0.0, color=False)
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(tour.run())
    render(tour.lines, Path(args.out))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size // 1024} KB, {len(tour.lines)} lines)")


if __name__ == "__main__":
    main()
