#!/usr/bin/env python3
"""Build a local-events-only src_rec catalog from the filtered 27-year file
(drops 'tel_' tele blocks and their cs pair lines).

Usage:
  python make_local_catalog.py [in_path out_path]
"""
import sys
from pathlib import Path

CASE = Path(__file__).parent.resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else CASE / "2_data_processing" / "src_rec_file_japan_2000_2023_filtered.dat"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else CASE / "2_data_processing" / "src_rec_file_japan_2000_2023_local.dat"

lines = SRC.read_text().splitlines()
hdr = [k for k, ln in enumerate(lines)
       if len(ln.split()) == 14 and not ln.split()[12].startswith("tel_")]
n_blocks = 0
n_lines = 0
with open(OUT, "w") as f:
    for i, h in enumerate(hdr):
        nxt = hdr[i + 1] if i + 1 < len(hdr) else len(lines)
        n_lines += (nxt - h)
        f.write("\n".join(lines[h:nxt]) + "\n")
        n_blocks += 1
print(f"wrote {OUT}: {n_blocks} local events in {n_lines} lines")
