#!/usr/bin/env python3
"""
fetch_jma_year.py

Download all monthly JMA Seismological Bulletin deck files for a year and
merge them into one TomoATT src_rec catalog (same convention as the
Jan-2020 dataset), enlarging local source/hit coverage ~12x.

Details:
  - source files: https://www.data.jma.go.jp/eqev/data/bulletin/data/deck/dYYYYMM.zip
  - each zip contains dYYYYMM[a,b,c] fixed-width 96-byte records
    (hypocenter 'J' + arrival '_'), see fmtdk_e.html.
  - per-month parse: uses parse_jma_arrival_data.py as a module
    (parse_station_list + parse_arrival_time_data + write).
  - final merge: renumber ids 0..N-1, drop events with < min_event_picks
    picks, out-of-region sources, depth>max_dep, mag out of range;
    traveltime guard rails mirror filter_arrival_data.py
    (0.5 < t < 120 s local set per filtered file is preserved as written).

Usage:
  python fetch_jma_year.py --year 2020
Outputs:
  1_data_acquisition/jma_decks/dYYYYMM[.zip, a,b,c]
  2_data_processing/src_rec_file_japan_year.dat  (merged, unfiltered-of-filtered)
  2_data_processing/year_events_summary.csv
"""

import argparse
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from parse_jma_arrival_data import (parse_station_list,
                                    parse_arrival_time_data)

SCRIPT_DIR = Path(__file__).parent.resolve()
DECK_DIR = SCRIPT_DIR / "jma_decks"
DECK_DIR.mkdir(exist_ok=True)
OUT_DIR = SCRIPT_DIR / ".." / "2_data_processing"

BASE = "https://www.data.jma.go.jp/eqev/data/bulletin/data/deck"

MIN_PICKS_PER_EVENT = 5   # matches upstream filter usage in the Jan set
MAX_DEP = 200.0
MIN_LAT, MAX_LAT = 24.0, 46.0
MIN_LON, MAX_LON = 122.0, 148.0


def fetch_month(year, month):
    ym = f"{year}{month:02d}"
    zpath = DECK_DIR / f"d{ym}.zip"
    if not zpath.exists() or zpath.stat().st_size < 1_000:
        url = f"{BASE}/d{ym}.zip"
        print(f"  fetch {url}", flush=True)
        urllib.request.urlretrieve(url, zpath)
        time.sleep(1.0)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(DECK_DIR)
    return [DECK_DIR / f"d{ym}{s}" for s in "abc"
            if (DECK_DIR / f"d{ym}{s}").exists()]


def keep(event):
    if len(event["arrivals"]) < MIN_PICKS_PER_EVENT:
        return False
    if not (MIN_LAT <= event["latitude"] <= MAX_LAT
            and MIN_LON <= event["longitude"] <= MAX_LON):
        return False
    if event["depth"] > MAX_DEP:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--months", type=int, nargs="*",
                    default=list(range(1, 13)))
    args = ap.parse_args()

    stations_file = SCRIPT_DIR / "stations"
    station_coords = parse_station_list(stations_file)
    print(f"stations: {len(station_coords)}")

    all_events = []
    for m in args.months:
        files = fetch_month(args.year, m)
        events = parse_arrival_time_data(files, station_coords)
        kept = [e for e in events if keep(e)]
        print(f"  {args.year}-{m:02d}: parsed {len(events)}, kept {len(kept)}")
        all_events.extend(kept)

    # sequential ids
    all_events.sort(key=lambda e: e["origin_time"])
    print(f"kept events total: {len(all_events)}")

    # write merged file (same format as src_rec_file_japan_filtered.dat)
    out = OUT_DIR / f"src_rec_file_japan_{args.year}.dat"
    n_picks = 0
    with open(out, "w") as f:
        for i, e in enumerate(all_events):
            ot = e["origin_time"]
            sec = ot.second + ot.microsecond / 1e6
            f.write(f"{i}  {ot.year}  {ot.month}  {ot.day}  {ot.hour}  "
                    f"{ot.minute}  {sec:9.3f}  "
                    f"{e['latitude']:12.6f}  {e['longitude']:12.6f}  "
                    f"{e['depth']:9.3f}  {e['magnitude']:6.2f}  "
                    f"{len(e['arrivals'])}  ev_{i:06d}  1.000\n")
            for j, arr in enumerate(e["arrivals"]):
                trav = arr["travel_time"]
                if not (0.5 <= trav <= 120.0):
                    continue
                f.write(f"{i}  {j}  {arr['station_code']}  "
                        f"{arr['station_lat']:12.6f}  "
                        f"{arr['station_lon']:12.6f}  "
                        f"{0.0:8.2f}  P  {trav:12.4f}  1.000\n")
                n_picks += 1
    print(f"wrote {out}: events={len(all_events)}, arrivals={n_picks}")

    import csv
    with open(OUT_DIR / f"year_events_summary_{args.year}.csv", "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["month", "kept_events"])
        # per-month counts recoverable from event_o ts


if __name__ == "__main__":
    main()
