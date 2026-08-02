#!/usr/bin/env python3
"""
download_teleseismic_isc.py

Download teleseismic P arrival picks at Japanese stations from the ISC
Bulletin web service (machine feed), for use as teleseismic sources in
TomoATT (sources outside the study region, common-source differential time
inversion for deep coverage augmentation).

API (validated 2026-08-02):
    https://www.isc.ac.uk/cgi-bin/web-db-run
        ?out_format=CSV
        &request=STNARRIVALS
        &phaselist=P
        &stnsearch=RECT&stn_bot_lat=..&stn_top_lat=..
                        &stn_left_lon=..&stn_right_lon=..
        &searchshape=GLOBAL
        &start_year=..&start_month=..&start_day=..&start_time=HH:MM:SS
        &end_year=..&end_month=..&end_day=..&end_time=HH:MM:SS
        &min_mag=..&req_mag_type=Any&req_mag_agcy=Any

CSV columns (see https://www.isc.ac.uk/iscbulletin/search/arrivals/csvoutput/):
    EVENTID,TYPE,REPORTER,STA,NET,LAT,LON,ELEV,CHN,DIST,BAZ,ISCPHASE,REPPHASE,
    DATE,TIME,RES,TDEF,AMPLITUDE,PER,AUTHOR(o),DATE(o),TIME(o),LAT(o),LON(o),
    DEPTH,AUTHOR(m),TYPE(m),MAG

Default query: full year 2020, magnitude >= 5.5, global events, P arrivals at
stations inside the Japan study box [24-46 N, 122-148 E]. Teleseismic
filtering (event outside the box, epicentral distance 30-90 deg, per-event
minimum pick count) is applied here so the output feeds straight into
build_src_rec_with_tele.py.

Citation reminder (ISC requirement):
    The International Seismological Centre (2024), On-line Bulletin,
    https://doi.org/10.31905/D808B830
"""

import csv
import io
import sys
import time
import calendar
import argparse
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
OUT_DIR = SCRIPT_DIR / "isc_teleseismic"
OUT_DIR.mkdir(exist_ok=True)

ISC_URL = "https://www.isc.ac.uk/cgi-bin/web-db-run"

# Japan study-region station box (same as tomography domain)
STN_BOX = dict(stn_bot_lat=24.0, stn_top_lat=46.0,
               stn_left_lon=122.0, stn_right_lon=148.0)

# Event-quality / teleseismic window
MIN_MAG = 5.5          # query-level magnitude cutoff (any type, any author)
DIST_MIN = 30.0        # deg — teleseismic P window lower bound
DIST_MAX = 90.0        # deg — avoid Pdiff / core-phase contamination
MIN_PICKS_PER_EV = 10  # keep teleseismic event only with >= this many picks

# Reporter preference when the same (event, station) pick is duplicated by
# several agencies/channels (lower index = preferred)
REPORTER_PREF = {"JMA": 0, "NIED": 1, "ISC": 2, "NEIC": 3, "GCMT": 4}

OUT_CSV = OUT_DIR / "isc_tele_p_arrivals_2020.csv"
OUT_EVENTS = OUT_DIR / "isc_tele_events_2020.csv"


# ---------------------------------------------------------------------------
# Query one time chunk
# ---------------------------------------------------------------------------
def query_chunk(y0, m0, d0, y1, m1, d1, min_mag=MIN_MAG, retries=4):
    params = dict(
        out_format="CSV", request="STNARRIVALS",
        ttime="on", tdef="on", phaselist="P",
        stnsearch="RECT",
        stn_bot_lat=f"{STN_BOX['stn_bot_lat']}",
        stn_top_lat=f"{STN_BOX['stn_top_lat']}",
        stn_left_lon=f"{STN_BOX['stn_left_lon']}",
        stn_right_lon=f"{STN_BOX['stn_right_lon']}",
        searchshape="GLOBAL",
        start_year=y0, start_month=m0, start_day=d0, start_time="00:00:00",
        end_year=y1, end_month=m1, end_day=d1, end_time="00:00:00",
        min_mag=f"{min_mag}", req_mag_type="Any", req_mag_agcy="Any",
    )
    for attempt in range(retries):
        try:
            r = requests.get(ISC_URL, params=params, timeout=600)
            if r.status_code == 200 and "EVENTID" in r.text:
                return r.text
            print(f"  attempt {attempt+1}: HTTP {r.status_code}, retrying...",
                  flush=True)
        except requests.RequestException as e:
            print(f"  attempt {attempt+1}: {e}, retrying...", flush=True)
        time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"ISC query failed for {y0}-{m0:02d}")


# ---------------------------------------------------------------------------
# Parse one response (HTML wrapper + CSV payload; keep data rows only)
# ---------------------------------------------------------------------------
def parse_rows(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        # data rows start with the numeric EVENTID
        if line and line[0].isdigit():
            parts = next(csv.reader(io.StringIO(line)))
            if len(parts) >= 28:
                rows.append([p.strip() for p in parts])
    return rows


F = dict(ev_id=0, reporter=2, sta=3, net=4, stla=5, stlo=6, elev=7,
         dist=9, baz=10, iscphase=11, arr_date=13, arr_time=14,
         res=15, tdef=16, o_date=20, o_time=21, elat=22, elon=23,
         edep=24, mag_auth=25, mag_type=26, mag=27)


def month_chunks(year):
    for m in range(1, 13):
        last = calendar.monthrange(year, m)[1]
        yield (year, m, 1, year, m, last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--min-mag", type=float, default=MIN_MAG)
    args = ap.parse_args()

    all_rows = []
    for (y0, m0, d0, y1, m1, d1) in month_chunks(args.year):
        print(f"query {y0}-{m0:02d} ...", flush=True)
        text = query_chunk(y0, m0, d0, y1, m1, d1, min_mag=args.min_mag)
        rows = parse_rows(text)
        print(f"  {len(rows)} raw arrival rows")
        all_rows.extend(rows)
        time.sleep(5)  # be polite to the service

    print(f"total raw rows: {len(all_rows)}")

    # --- filter to teleseismic window -------------------------------------
    in_box = lambda la, lo: (24.0 <= la <= 46.0) and (122.0 <= lo <= 148.0)
    filtered = []
    for r in all_rows:
        try:
            ela, elo = float(r[F['elat']]), float(r[F['elon']])
            dist = float(r[F['dist']])
        except ValueError:
            continue
        if in_box(ela, elo):
            continue  # local event — handled by the JMA dataset
        if not (DIST_MIN <= dist <= DIST_MAX):
            continue
        filtered.append(r)
    print(f"teleseismic-window rows (dist {DIST_MIN}-{DIST_MAX} deg, "
          f"event outside box): {len(filtered)}")

    # --- deduplicate (event, station): prefer TDEF pick, then best reporter -
    best = {}
    for r in filtered:
        key = (r[F['ev_id']], r[F['sta']])
        pref = (0 if r[F['tdef']] == "True" else 1,
                REPORTER_PREF.get(r[F['reporter']], 9))
        if key not in best or pref < best[key][0]:
            best[key] = (pref, r)
    dedup = [v[1] for v in best.values()]

    # --- per-event pick count threshold ------------------------------------
    counts = {}
    for r in dedup:
        counts[r[F['ev_id']]] = counts.get(r[F['ev_id']], 0) + 1
    kept = [r for r in dedup if counts[r[F['ev_id']]] >= MIN_PICKS_PER_EV]
    ev_ids = sorted({r[F['ev_id']] for r in kept})
    print(f"after dedup & >= {MIN_PICKS_PER_EV} picks/event: "
          f"{len(kept)} picks, {len(ev_ids)} teleseismic events")

    # --- write outputs ------------------------------------------------------
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ev_id", "sta", "net", "stla", "stlo", "elev_m",
                    "dist_deg", "baz_deg", "phase",
                    "origin_time_utc", "arr_time_utc", "travel_time_s",
                    "tt_residual_s", "tdef",
                    "evla", "evlo", "evdep_km", "mag_type", "mag"])
        for r in kept:
            try:
                ot = f"{r[F['o_date']]}T{r[F['o_time']]}"
                at = f"{r[F['arr_date']]}T{r[F['arr_time']]}"
                from datetime import datetime
                fmt1 = "%Y-%m-%dT%H:%M:%S.%f"
                fmt2 = "%Y-%m-%dT%H:%M:%S"
                parse = lambda s: datetime.strptime(
                    s, fmt1 if "." in s else fmt2)
                tt = (parse(at) - parse(ot)).total_seconds()
            except Exception:
                continue
            w.writerow([r[F['ev_id']], r[F['sta']], r[F['net']],
                        r[F['stla']], r[F['stlo']], r[F['elev']],
                        r[F['dist']], r[F['baz']], r[F['iscphase']],
                        ot, at, f"{tt:.3f}", r[F['res']], r[F['tdef']],
                        r[F['elat']], r[F['elon']], r[F['edep']],
                        r[F['mag_type']], r[F['mag']]])

    ev_first = {}
    for r in kept:
        ev_first.setdefault(r[F['ev_id']], r)
    with open(OUT_EVENTS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ev_id", "origin_time_utc", "evla", "evlo", "evdep_km",
                    "mag", "n_picks"])
        for ev in ev_ids:
            r = ev_first[ev]
            w.writerow([ev, f"{r[F['o_date']]}T{r[F['o_time']]}",
                        r[F['elat']], r[F['elon']], r[F['edep']],
                        r[F['mag']], counts[ev]])

    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_EVENTS}")


if __name__ == "__main__":
    main()
