"""
Parse fine-grained class folder names (combined_data) into (crop, disease) for hierarchical training.
"""

from __future__ import annotations

import re


def parse_fine_class_to_crop_disease(fine_class: str) -> tuple[str, str]:
    """
    Map one training class folder name to (crop_bucket, disease_bucket).

    EXT_* folders are built as EXT_<prefix>_<sanitized_class> where prefix may be
    one token (mini) or two (rice_vbk). We try a two-token prefix first, then one.
    """
    name = fine_class.strip()
    if not name:
        return "Unknown", "unknown"

    if name.startswith("EXT_"):
        rest = name[4:]
        m = re.match(r"^([^_]+_[^_]+)_(.+)$", rest)
        if m:
            return m.group(1), m.group(2)
        m2 = re.match(r"^([^_]+)_(.+)$", rest)
        if m2:
            return m2.group(1), m2.group(2)
        return rest, "class"

    if name.startswith("CPDD_"):
        m = re.match(r"^CPDD_([^_]+)_(.+)$", name)
        if m:
            return m.group(1), m.group(2)
        return name[5:], "class"

    if "___" in name:
        crop, disease = name.split("___", 1)
        return crop.strip(), disease.strip()

    return "Unknown", name


def sanitize_fs_segment(s: str) -> str:
    """Safe single path segment (Windows-friendly)."""
    s = re.sub(r'[<>:"/\\|?*]', "_", s.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"
