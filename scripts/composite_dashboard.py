"""Composite four dashboard panel screenshots into a 2x2 grid for the README.

Usage (Pillow comes with the dashboard extra):
    uv run --extra dashboard python scripts/composite_dashboard.py \\
        docs/_tab1.png docs/_tab2.png docs/_tab3.png docs/_tab4.png \\
        docs/dashboard.png

Expects four PNGs of similar dimensions. Resizes each to a uniform tile width
and stacks them in a labelled 2x2 grid with thin separators.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Tile width per panel in the final composite. Two tiles wide + a small
# separator gives a final image around 2400px: readable on GitHub at the
# 800px display width because the source is high-DPI to begin with.
TILE_WIDTH = 1200
TILE_GAP = 8
OUTER_BORDER = 4
LABEL_HEIGHT = 36
PAGE_COLOR = (255, 255, 255)
"""Streamlit's light-theme page background, the boundary both crops look for."""
TOOLBAR_SCAN_WIDTH_RATIO = 0.8
"""Content-top search ignores the right 20%, where the Deploy toolbar sits."""
TOP_MARGIN_RATIO = 0.02
"""White space kept above the first content row, as a share of the image height."""
BACKGROUND_COLOR = (255, 255, 255)
SEPARATOR_COLOR = (220, 220, 220)
LABEL_COLOR = (40, 40, 40)


def _load_font() -> ImageFont.ImageFont:
    """First macOS system font found, else Pillow's built-in bitmap font."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, 22)
            except OSError:
                continue
    return ImageFont.load_default()


def _sidebar_right_edge(source: Image.Image) -> int:
    """First column of the page background on a row below the sidebar widgets.

    Measured, not fixed: the sidebar's pixel width depends on the screen scale
    and the window, and it shows the local DB path. Returns 0 when it is collapsed.
    """
    row = int(source.height * 0.75)
    for x in range(source.width // 2):
        if source.getpixel((x, row)) == PAGE_COLOR:
            return x
    return 0


def _content_top(source: Image.Image) -> int:
    """First row holding content, minus a margin, so the Deploy toolbar is cut off."""
    scan = source.crop((0, 0, int(source.width * TOOLBAR_SCAN_WIDTH_RATIO), source.height))
    for y in range(scan.height):
        row = scan.crop((0, y, scan.width, y + 1))
        if row.getcolors(maxcolors=1) != [(scan.width, PAGE_COLOR)]:
            return max(0, y - int(source.height * TOP_MARGIN_RATIO))
    return 0


def _tile(image_path: str, label: str, font: ImageFont.ImageFont) -> Image.Image:
    """One panel: sidebar and toolbar cropped, resized to ``TILE_WIDTH``, under a label."""
    source = Image.open(image_path).convert("RGB")
    source = source.crop((_sidebar_right_edge(source), 0, source.width, source.height))
    source = source.crop((0, _content_top(source), source.width, source.height))
    scale = TILE_WIDTH / source.width
    panel = source.resize((TILE_WIDTH, int(source.height * scale)), Image.LANCZOS)

    tile = Image.new("RGB", (TILE_WIDTH, LABEL_HEIGHT + panel.height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(tile)
    draw.text((16, 6), label, fill=LABEL_COLOR, font=font)
    draw.line(
        [(0, LABEL_HEIGHT - 1), (TILE_WIDTH, LABEL_HEIGHT - 1)],
        fill=SEPARATOR_COLOR,
        width=1,
    )
    tile.paste(panel, (0, LABEL_HEIGHT))
    return tile


def main(paths: list[str], labels: list[str], out_path: str) -> None:
    """Write a labelled 2x2 grid of the four panels to ``out_path``."""
    if len(paths) != 4:
        raise SystemExit(f"expected 4 input images, got {len(paths)}")
    font = _load_font()
    tiles = [_tile(path, label, font) for path, label in zip(paths, labels, strict=True)]

    # Pad every tile to the tallest so the grid lines up.
    tile_height = max(tile.height for tile in tiles)
    padded_tiles: list[Image.Image] = []
    for tile in tiles:
        if tile.height == tile_height:
            padded_tiles.append(tile)
            continue
        padded = Image.new("RGB", (TILE_WIDTH, tile_height), BACKGROUND_COLOR)
        padded.paste(tile, (0, 0))
        padded_tiles.append(padded)

    final_width = TILE_WIDTH * 2 + TILE_GAP + OUTER_BORDER * 2
    final_height = tile_height * 2 + TILE_GAP + OUTER_BORDER * 2
    final = Image.new("RGB", (final_width, final_height), SEPARATOR_COLOR)
    right_x = OUTER_BORDER + TILE_WIDTH + TILE_GAP
    bottom_y = OUTER_BORDER + tile_height + TILE_GAP
    final.paste(padded_tiles[0], (OUTER_BORDER, OUTER_BORDER))
    final.paste(padded_tiles[1], (right_x, OUTER_BORDER))
    final.paste(padded_tiles[2], (OUTER_BORDER, bottom_y))
    final.paste(padded_tiles[3], (right_x, bottom_y))

    final.save(out_path, optimize=True)
    print(f"wrote {out_path} ({final_width}x{final_height})")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 5:
        raise SystemExit(
            "usage: composite_dashboard.py panel1.png panel2.png panel3.png panel4.png out.png"
        )
    labels = [
        "Cost savings",
        "Routing accuracy",
        "Corrections",
        "Model comparison",
    ]
    main(args[:4], labels, args[4])
