# Visual assets

- `distributed-wordmark.png` is the supplied Distributed Systems textured copper wordmark, copied unchanged from the existing brand kit. Keep its aspect ratio, letterforms and clear space. Copper reference: `#E2A674`.
- `hero.png` (1792 × 896, rendered at 3584 × 1792 and halved) is illustrative launch artwork, not a screenshot of the product. Its sources are in `hero-src/` and reproduce it exactly.
- `workflow.svg` is a repository-native vector explanation of the implemented fetch, transcribe, frame and question pipeline. It describes the tool contract; it is not a benchmark.

## How the hero is made

1. **Artwork** (`hero-src/art.jpg`). Generated with OpenAI `gpt-image-2`, then recomposed with the images **edit** endpoint (`hero-src/edit.py`, prompt in `hero-src/edit-prompt.txt`). The recomposition moved the video window below the headline area, so the type and the scene interleave instead of competing for width. The render contains no text, grid, marks or frame.
2. **Composition** (`python hero-src/compose.py hero-src/art.jpg distributed-wordmark.png out.png`). Everything drawn sits on one square grid: pitch 128 px, margin 64 px, 27 × 13 cells at 3584 × 1792.

| Element | Grid position |
| --- | --- |
| Corner crosshairs | the four outer grid corners |
| Frame | top line between the top crosshairs; left line down the margin; bottom line stops at column 8, where the plinth breaks the frame |
| Headline (Avenir Next Heavy) | cap height exactly 1 cell; cap top on row 2; baselines on rows 3, 4.5 and 6; left edge on column 1 |
| Rules | rows 7 and 12, column 0 to column 8, with crosshairs on the frame |
| Wordmark | visible letters exactly row 8 to row 9 (one cell tall) |
| Mono lines (JetBrains Mono) | baselines on rows 10.5 and 11 |
| Tick strip | columns 24 to 26, rows 12.5 to 12.75 |
| Dashed arc | concentric with the copper disc (centre and radius fitted to its rim, median residual 0.44 px), 46 px outside it, starting below the top frame |

The grid, frame lines and arc are masked by the artwork's brightness, so they pass behind the objects. `compose.py` checks its own layout on every run: it reports any artwork pixel inside a 48 px halo around the type (currently zero) and the answer card's clearance from the right frame (55 px). The fonts are macOS system fonts plus JetBrains Mono; on another machine, set the font paths at the top of the script.

The Distributed Systems name and wordmark remain brand assets. Inclusion does not grant trademark rights or imply endorsement of derivative projects.
