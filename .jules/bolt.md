## 2025-02-12 - [Redundant Computation in Text Rendering]
**Learning:** `draw.textbbox()` in Python's PIL (Pillow) library was being called redundantly in `comic_builder.py` - multiple times per line for calculating widths, heights, and finally drawing text. This was a noticeable performance bottleneck during the text generation loop.
**Action:** Always cache bounding box dimensions (`width` and `height`) using a single `draw.textbbox()` call per text element rather than re-computing them repeatedly when rendering multi-line text.
