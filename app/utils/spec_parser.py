import math
import numpy as np

# --- Pre-data --- 
def _clean_spec_pred(spec_pred):
    """
    Normalize spec_pred:
    - return None for NaN / None / empty
    - return stripped string otherwise
    """
    # None or NaN (float)
    if spec_pred is None:
        return None
    if isinstance(spec_pred, float) and math.isnan(spec_pred):
        return None

    # Convert to string and strip
    s = str(spec_pred).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s


def parse_spec(spec_pred):
    """
    Convert spec string like:
        'application -|brand sanden|item ตู้แช่|model tr25'
    into a dict:
        {'application': '-', 'brand': 'sanden', 'item': 'ตู้แช่', 'model': 'tr25'}
    Handles NaN / None safely by returning {}.
    """
    spec_str = _clean_spec_pred(spec_pred)
    if not spec_str:
        return {}
    # print(spec_str)

    pairs = spec_str.split("|")
    spec_dict = {}
    for p in pairs:
        p = p.strip()
        if not p:
            continue
        parts = p.split(" ", 1)
        if len(parts) != 2:
            continue
        key, val = parts
        spec_dict[key] = val
    return spec_dict


def extract_item(spec_pred):
    """
    Extract 'item' value from spec_pred.
    - If not present or '-' or empty -> return NaN
    """
    spec = parse_spec(spec_pred)
    item = spec.get("item")

    if not item or item.strip() == "-" or item.strip() == "":
        return np.nan
    return item


# --- After run LLM ---
# Make sure each key-value have one internal space and always lowercase (Pydantic)
def fix_spec_format(spec_string: str) -> str:
    """Remove internal spaces from keys and values while preserving key-value separation
       Convert to lowercase"""
    if not spec_string:
        return spec_string

    pairs = spec_string.split("|")
    fixed_pairs = []

    for pair in pairs:
        parts = pair.strip().split(" ", 1)  # Split only on first space
        if len(parts) == 2:
            key = parts[0].replace(" ", "").lower()
            value = parts[1].replace(" ", "").lower()
            fixed_pairs.append(f"{key} {value}")

    return "|".join(fixed_pairs)


def clean_missing_values(spec_string: str) -> str:
    """
    Replace any '?' value with '-' AFTER final normalization.
    """
    normalized = fix_spec_format(spec_string or "")

    if not normalized:
        return normalized

    parts = normalized.split("|")
    out = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        key_value = part.split(" ", 1)
        if len(key_value) != 2:
            continue

        key, value = key_value

        if value == "?":
            value = "-"

        out.append(f"{key} {value}")

    return "|".join(out)


def align_spec_keys(original_spec: str, fixed_spec: str) -> str:
    """
    Ensure fixed_spec contains ONLY keys from original_spec, in the same order.
    Missing keys in fixed_spec will be filled with "-" from original schema.
    Extra keys in fixed_spec will be removed.
    """
    original_dict = parse_spec(original_spec)
    if not original_dict:
        return ""

    fixed_dict = parse_spec(fixed_spec)

    aligned_parts = []
    for key in original_dict.keys():
        # Use value from fixed_spec if exists, otherwise "-"
        val = fixed_dict.get(key, "-")
        aligned_parts.append(f"{key} {val}")
        
    return "|".join(aligned_parts)

