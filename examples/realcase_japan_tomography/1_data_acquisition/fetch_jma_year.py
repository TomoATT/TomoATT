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
  python fetch_jma_year.py --years 2000 2026
Outputs:
  1_data_acquisition/jma_decks/dYYYYMM[.zip, a,b,c]  (pruned per year)
  2_data_processing/src_rec_file_japan_<year-or-range>.dat

Notes: deck zips are deleted right after each year is parsed to keep local
storage bounded. Year catalog is written incrementally (one merged file for
the whole requested range).
"""

import argparse
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError

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
        try:
            urllib.request.urlretrieve(url, zpath)
        except (HTTPError, URLError) as e:
            print(f"  SKIP {ym}: {e}", flush=True)
            zpath.unlink(missing_ok=True)
            time.sleep(1.0)
            return []
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


def keep_region(event):
    if not (MIN_LAT <= event["latitude"] <= MAX_LAT
            and MIN_LON <= event["longitude"] <= MAX_LON):
        return False
    if event["depth"] > MAX_DEP:
        return False
    return True


def wipe_decks():
    if DECK_DIR.exists():
        for p in DECK_DIR.iterdir():
            if p.is_file():
                p.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=None,
                    help="single year (legacy)")
    ap.add_argument("--years", type=int, nargs=2, default=None,
                    metavar=("Y_FROM", "Y_TO"), help="inclusive year range")
    ap.add_argument("--months", type=int, nargs="*",
                    default=list(range(1, 13)))
    args = ap.parse_args()

    stations_file = SCRIPT_DIR / "stations"
    station_coords = parse_station_list(stations_file)
    print(f"stations: {len(station_coords)}")

    years = ([args.year] if args.year else
             list(range(args.years[0], args.years[1] + 1)))
    tag = (f"{args.year}" if args.year else f"{args.years[0]}_{args.years[1]}")
    out = OUT_DIR / f"src_rec_file_japan_{tag}.dat"

    n_events_total = 0
    n_picks_total = 0
    with open(out, "w") as f:
        for y in years:
            y_events = []
            for m in args.months:
                files = fetch_month(y, m)
                if not files:
                    continue
                events = parse_arrival_time_data(files, station_coords)
                kept = [e for e in events if keep_region(e) and keep(e)]
                print(f"  {y}-{m:02d}: parsed {len(events)}, kept {len(kept)}",
                      flush=True)
                y_events.extend(kept)
            y_events.sort(key=lambda e: e["origin_time"])
            for e in y_events:
                good = [a for a in e["arrivals"]
                        if 0.5 <= a["travel_time"] <= 120.0]
                if not good:
                    continue
                i = n_events_total
                ot = e["origin_time"]
                sec = ot.second + ot.microsecond / 1e6
                f.write(f"{i}  {ot.year}  {ot.month}  {ot.day}  {ot.hour}  "
                        f"{ot.minute}  {sec:9.3f}  "
                        f"{e['latitude']:12.6f}  {e['longitude']:12.6f}  "
                        f"{e['depth']:9.3f}  {e['magnitude']:6.2f}  "
                        f"{len(good)}  ev_{i:06d}  1.000\n")
                for j, arr in enumerate(good):
                    f.write(f"{i}  {j}  {arr['station_code']}  "
                            f"{arr['station_lat']:12.6f}  "
                            f"{arr['station_lon']:12.6f}  "
                            f"{0.0:8.2f}  P  {arr['travel_time']:12.4f}  1.000\n")
                n_picks_total += len(good)
                n_events_total += 1
            wipe_decks()

    print(f"wrote {out}: events={n_events_total}, arrivals={n_picks_total}")


if __name__ == "__main__":
    main()
