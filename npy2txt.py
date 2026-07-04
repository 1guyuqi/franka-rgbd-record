#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def format_obj(obj) -> str:
    if isinstance(obj, np.ndarray):

        if obj.dtype == object:

            if obj.shape == ():
                return format_obj(obj.item())

            return "\n".join(f"[{i}] {format_obj(x)}" for i, x in enumerate(obj.tolist()))

        return np.array2string(obj, separator=", ", threshold=np.inf, max_line_width=10**9)

    if isinstance(obj, (np.generic,)):
        return str(obj.item())

    if isinstance(obj, dict):
        lines = []
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            lines.append(f"{k}: {format_obj(obj[k])}")
        return "\n".join(lines)

    if isinstance(obj, (list, tuple)):
        return "\n".join(f"[{i}] {format_obj(x)}" for i, x in enumerate(obj))

    return repr(obj)


def main():
    parser = argparse.ArgumentParser(description="Load a .npy file, print all contents, and export to .txt")
    parser.add_argument("npy_path", type=str, help="Path to the .npy file")
    parser.add_argument("-o", "--out", type=str, default=None, help="Output .txt path (default: same name as .npy)")
    args = parser.parse_args()

    npy_path = Path(args.npy_path).expanduser().resolve()
    if not npy_path.exists():
        raise FileNotFoundError(f"Not found: {npy_path}")

    out_path = Path(args.out).expanduser().resolve() if args.out else npy_path.with_suffix(".txt")

    data = np.load(npy_path, allow_pickle=True)

    text = format_obj(data)

    print(text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"\n[OK] Saved to: {out_path}")


if __name__ == "__main__":
    main()