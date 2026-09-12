"""Generate a 1200x630 thumbnail.png in every folder (nested included) that has an HTML page.

Usage: python _scripts/generate_thumbnail.py
"""
import html
import os
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
AUTHOR = "Aditya Ramesh"

WIDTH, HEIGHT = 1200, 630
MARGIN = 72
BG = "#ffffff"
TITLE_COLOR = "#1f1c19"
AUTHOR_COLOR = "#1f1c19"

SANS = ["Roboto-Light.ttf", "Roboto-Regular.ttf", "DejaVuSans.ttf"]
SANS_BOLD = ["Roboto-Medium.ttf", "Roboto-Bold.ttf", "DejaVuSans-Bold.ttf"]

FONT_DIRS = [
    "C:/Windows/Fonts",
    os.path.expandvars("%LOCALAPPDATA%/Microsoft/Windows/Fonts"),
    "/Library/Fonts",
    os.path.expanduser("~/Library/Fonts"),
    "/usr/share/fonts/truetype/roboto",
]


def load_font(names, size):
    for name in names:
        for candidate in [name] + [f"{d}/{name}" for d in FONT_DIRS]:
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def page_title(page):
    """Title from <title>, then <h1>, then a sibling Markdown '# ' heading, then the folder name."""
    text = page.read_text(encoding="utf-8")
    title = None
    for pattern in (r"<title[^>]*>(.*?)</title>", r"<h1[^>]*>(.*?)</h1>"):
        match = re.search(pattern, text, re.S | re.I)
        if match:
            title = html.unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip()
            if title:
                break
    if not title:
        for md in sorted(page.parent.glob("*.md")):
            match = re.search(r"^#\s+(.+)$", md.read_text(encoding="utf-8"), re.M)
            if match:
                title = match.group(1).strip()
                break
    title = title or page.parent.name
    # Drop a trailing " — Aditya Ramesh" / " | Aditya Ramesh" suffix.
    title = re.sub(rf"\s*[—|\-–:]\s*{re.escape(AUTHOR)}\s*$", "", title)
    return " ".join(title.split())


def wrap(draw, text, font, max_width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def balance(draw, text, font, max_width):
    """Wrap, then even out line lengths without adding lines."""
    lines = wrap(draw, text, font, max_width)
    count = len(lines)
    best = lines
    lo, hi = 1, max_width
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = wrap(draw, text, font, mid)
        if len(trial) <= count and all(draw.textlength(l, font=font) <= mid for l in trial):
            best, hi = trial, mid - 1
        else:
            lo = mid + 1
    return best


def render(title, out_path):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    max_width = int((WIDTH - 2 * MARGIN) * 0.85)

    # Shrink the title until it fits in at most three lines.
    for size in range(92, 47, -4):
        title_font = load_font(SANS, size)
        if len(wrap(draw, title, title_font, max_width)) <= 3:
            break
    lines = balance(draw, title, title_font, max_width)
    author_font = load_font(SANS_BOLD, max(44, int(title_font.size * 0.6)))
    show_author = title != AUTHOR
    if not show_author:
        title_font = load_font(SANS_BOLD, title_font.size)

    # Stack from the bottom up, anchored on baselines, with tight leading.
    line_height = int(title_font.size * 1.08)
    baseline = HEIGHT - MARGIN
    if show_author:
        draw.text((MARGIN, baseline), AUTHOR, font=author_font, fill=AUTHOR_COLOR, anchor="ls")
        baseline -= int(author_font.size * 1.45)
    for line in reversed(lines):
        draw.text((MARGIN, baseline), line, font=title_font, fill=TITLE_COLOR, anchor="ls")
        baseline -= line_height

    img.save(out_path)


def main():
    pages = {}
    for html_file in sorted(ROOT.rglob("*.html")):
        rel = html_file.relative_to(ROOT)
        if any(part.startswith((".", "_")) for part in rel.parts):
            continue
        # One thumbnail per folder: prefer index.html, else the first page found.
        if html_file.parent not in pages or html_file.name == "index.html":
            pages[html_file.parent] = html_file

    for folder, page in sorted(pages.items()):
        title = page_title(page)
        out_path = folder / "thumbnail.png"
        render(title, out_path)
        print(f"{out_path.relative_to(ROOT)}  <-  {title}")


if __name__ == "__main__":
    main()
