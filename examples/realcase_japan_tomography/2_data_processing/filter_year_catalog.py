#!/usr/bin/env python3
"""
filter_year_catalog.py

Turn the raw full-year JMA 2020 parse (src_rec_file_japan_2020.dat)
into the production-grade catalog for the v3 inversion, then REMOVE the
raw downloaded JMA deck files to conserve local storage.

QC stages (counts printed at each stage):
  1. travel-time window: 0.5 <= t <= 120 s
  2. per-event redundancy: >= 5 usable picks
  3. station floor: stations with fewer than STATION_MIN_PICKS total picks
     are removed (platforms/retired/navigation stations cannot form
     coherent path statistics)
  4. distance-consistent traveltime window: for straight-line epicentral
     distance d (km), keep arrivals with  d/VMAX_S <= t <= d/VMIN + 15 s,
     VMAX=9.2 km/s, VMIN=6.0 km/s (kills cross-region S picks and
     unrealistically fast/slow ones)
  5. drop duplicate (event,station) picks (keep first)

Outputs:
  src_rec_file_japan_2020_filtered.dat
  year_filter_stats.csv  (per-stage counts)

Cleanup: remove 1_data_acquisition/jma_decks/* (zips + extracted records).
"""

import math
import sys
from pathlib import Path
from collections import defaultdict

import argparse

CASE = Path(__file__).parent.resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--in", dest="src", default=str(CASE / "2_data_processing" / "src_rec_file_japan_2020.dat"))
p.add_argument("--out", dest="out", default=str(CASE / "2_data_processing" / "src_rec_file_japan_2020_filtered.dat"))
p.add_argument("--stats", dest="stats", default=str(CASE / "2_data_processing" / "year_filter_stats.csv"))
args = p.parse_args()
SRC = Path(args.src)
OUT = Path(args.out)
STATS = Path(args.stats)
DECKS = CASE / "1_data_acquisition" / "jma_decks"

MIN_LAT, MAX_LAT = 24.0, 46.0
MIN_LON, MAX_LON = 122.0, 148.0
MAX_DEP = 200.0
MIN_PICKS_PER_EVENT = 5
STATION_MIN_PICKS = 50
T_MIN, T_MAX = 0.5, 120.0
V_FAST, V_SLOW, SLOW_PAD = 9.2, 6.0, 15.0


def gc_km(lat1, lon1, lat2, lon2):
    d = math.pi / 180.0
    a = (math.sin(lat1 * d) * math.sin(lat2 * d)
         + math.cos(lat1 * d) * math.cos(lat2 * d)
         * math.cos((lon2 - lon1) * d))
    return 6371.0 * math.acos(max(-1.0, min(1.0, a)))


def load(path):
    lines = path.read_text().splitlines()
    idx, events = [k for k, ln in enumerate(lines)
                   if len(ln.split()) == 14], None
    ev = []
    for n, h in enumerate(idx):
        t = lines[h].split()
        nxt = idx[n + 1] if n + 1 < len(idx) else len(lines)
        recs = []
        for ln in lines[h + 1:nxt]:
            r = ln.split()
            if len(r) == 9:
                recs.append((r[2], float(r[3]), float(r[4]),
                             float(r[7])))
        ev.append(dict(hdr=t, recs=recs))
    return ev


def main():
    events = load(SRC)
    stats = []

    # --- 1+2: per-pick time window + per-event pick count ----------------
    out = []
    for e in events:
        d = float(e["hdr"][9])
        la, lo = float(e["hdr"][7]), float(e["hdr"][8])
        if not (MIN_LAT <= la <= MAX_LAT and MIN_LON <= lo <= MAX_LON
                and d <= MAX_DEP):
            continue
        recs = [r for r in e["recs"] if T_MIN <= r[3] <= T_MAX]
        if len(recs) >= MIN_PICKS_PER_EVENT:
            e2 = dict(e); e2["recs"] = recs
            out.append(e2)
    stats.append(("after time/region/depth/picks", len(out),
                  sum(len(e["recs"]) for e in out)))

    # --- 5: dedupe (event, station) --------------------------------------
    for e in out:
        seen, recs = set(), []
        for r in e["recs"]:
            if r[0] in seen:
                continue
            seen.add(r[0]); recs.append(r)
        e["recs"] = recs
    out = [e for e in out if len(e["recs"]) >= MIN_PICKS_PER_EVENT]
    stats.append(("after dedupe + picks recheck", len(out),
                  sum(len(e["recs"]) for e in out)))

    # --- 3: station floor -------------------------------------------------
    station_count = defaultdict(int)
    for e in out:
        for r in e["recs"]:
            station_count[r[0]] += 1
    keep_sta = {s for s, n in station_count.items() if n >= STATION_MIN_PICKS}
    out2 = []
    for e in out:
        e["recs"] = [r for r in e["recs"] if r[0] in keep_sta]
        if len(e["recs"]) >= MIN_PICKS_PER_EVENT:
            out2.append(e)
    out = out2
    stats.append(("after station floor", len(out),
                  sum(len(e["recs"]) for e in out)))

    # --- 4: distance-consistent traveltimes --------------------------------
    out2 = []
    for e in out:
        evla, evlo, evdp = (float(e["hdr"][7]), float(e["hdr"][8]),
                            float(e["hdr"][9]))
        dxy = gc_km(evla, evlo, evla, evlo)  # placeholder overwritten below
        recs = []
        for r in e["recs"]:
            d = gc_km(evla, evlo, r[1], r[2])
            d_eff = math.hypot(d, evdp)      # include depth (approx innate)
            low = d_eff / V_FAST
            high = d_eff / V_SLOW + SLOW_PAD
            if low <= r[3] <= high:
                recs.append(r)
        e["recs"] = recs
        if len(recs) >= MIN_PICKS_PER_EVENT:
            out2.append(e)
    out = out2
    stats.append(("after distance-window QC", len(out),
                  sum(len(e["recs"]) for e in out)))

    # --- write -------------------------------------------------------------
    n_picks = 0
    with open(OUT, "w") as f:
        for i, e in enumerate(out):
            h = e["hdr"]
            f.write(f"{i}  {h[1]}  {h[2]}  {h[3]}  {h[4]}  {h[5]}  {h[6]}  "
                    f"{h[7]}  {h[8]}  {h[9]}  {h[10]}  {len(e['recs'])}  "
                    f"{h[12]}  1.000\n")
            for j, r in enumerate(e["recs"]):
                f.write(f"{i}  {j}  {r[0]}  {r[1]:12.6f}  {r[2]:12.6f}  "
                        f"{0.0:8.2f}  P  {r[3]:12.4f}  1.000\n")
                n_picks += 1

    with open(STATS, "w") as f:
        f.write("stage,events,arrivals\n")
        for s in stats:
            f.write(f"{s[0]},{s[1]},{s[2]}\n")
        f.write(f"unique stations,{len(station_count)},\n")
        f.write(f"stations kept,{len(keep_sta)},\n")

    print("QC funnel:")
    for s in stats:
        print(f"  {s[0]:32s}  {s[1]:>8,d} events   {s[2]:>10,d} arrivals")
    print(f"stations: {len(station_count)} seen, {len(keep_sta)} kept")
    print(f"final: {len(out)} events, {n_picks} arrivals -> {OUT}")

    # --- cleanup decks ------------------------------------------------------
    if DECKS.exists():
        removed = 0
        for p in DECKS.iterdir():
            if p.is_file():
                p.unlink(); removed += 1
        print(f"deleted {removed} raw deck files from {DECKS}")


if __name__ == "__main__":
    main()
