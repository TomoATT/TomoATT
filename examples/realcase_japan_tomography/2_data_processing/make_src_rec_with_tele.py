#!/usr/bin/env python3
"""
build_src_rec_with_tele.py

Merge the regional (JMA/Hi-net, Jan-2020) src_rec file with the ISC-based
teleseismic source set into one TomoATT src_rec file for joint regional +
teleseismic inversion.

  - local events (inside the box, abs_time data): kept unchanged from
    src_rec_file_japan_filtered.dat
  - teleseismic events (outside the box, cs_dif_time data): appended as
    additional source blocks with receivers = Japanese stations and
    travel_time = ISC arrival time - ISC prime origin time.

TomoATT treats any source located outside the study region as teleseismic
when `have_tele_data: true` is set in the input params; the common-source
differential time inversion (use_cs_time: true) is then applied to those
blocks so that origin-time / outside-region path errors cancel.

Inputs:
  src_rec_file_japan_filtered.dat           (regional dataset)
  ../1_data_acquisition/isc_teleseismic/isc_tele_p_arrivals_2020.csv

Output:
  src_rec_file_japan_tele.dat               (merged file)
  tele_station_summary.csv                  (tele receiver inventory)
"""

import csv
from datetime import datetime
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent.resolve()
LOCAL_SRC_REC = SCRIPT_DIR / "src_rec_file_japan_filtered.dat"
TELE_CSV = (SCRIPT_DIR / ".." / "1_data_acquisition" /
            "isc_teleseismic" / "isc_tele_p_arrivals_2020.csv")
OUT_FILE = SCRIPT_DIR / "src_rec_file_japan_tele.dat"
OUT_STA = SCRIPT_DIR / "tele_station_summary.csv"

DEFAULT_MAG = 5.5   # ISC rows with empty magnitude column get this nominal value


def parse_dt(date_str, time_str):
    s = f"{date_str} {time_str}"
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f") if "." in s \
        else datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def main():
    # ---------------- load tele arrivals ----------------------------------
    events = defaultdict(list)   # ev_id -> list of pick dicts
    with open(TELE_CSV) as f:
        for r in csv.DictReader(f):
            events[r["ev_id"]].append(r)

    # ---------------- write merged file -----------------------------------
    with open(LOCAL_SRC_REC) as fin, open(OUT_FILE, "w") as fout:
        local_text = fin.read()
        fout.write(local_text.rstrip() + "\n")

        # count local event blocks (event lines have 14 columns; receiver
        # lines have 9) so tele source ids do not collide with local ones
        id_offset = sum(1 for ln in local_text.splitlines()
                        if len(ln.split()) == 14)
        print(f"local event blocks   : {id_offset}")

        n_events = len(events)
        rec_count = 0
        station_set = set()
        for k, (ev_id, picks) in enumerate(sorted(events.items())):
            i = id_offset + k
            ev_p0 = picks[0]
            evla = float(ev_p0["evla"])
            evlo = float(ev_p0["evlo"])
            evdep = float(ev_p0["evdep_km"])
            mag = float(ev_p0["mag"]) if ev_p0["mag"] else DEFAULT_MAG
            to = parse_dt(*ev_p0["origin_time_utc"].split("T"))
            sec = to.second + to.microsecond / 1e6

            fout.write(
                f"{i}  {to.year}  {to.month}  {to.day}  {to.hour}  "
                f"{to.minute}  {sec:9.3f}  "
                f"{evla:12.6f}  {evlo:12.6f}  {evdep:9.3f}  "
                f"{mag:6.2f}  {len(picks)}  tel_{ev_id}  1.000\n"
            )
            for j, p in enumerate(picks):
                stla = float(p["stla"])
                stlo = float(p["stlo"])
                fout.write(
                    f"{j}  {i}  {p['sta']}  {stla:12.6f}  {stlo:12.6f}  "
                    f"{0.0:8.2f}  P  {float(p['travel_time_s']):12.4f}  1.000\n"
                )
                station_set.add((p["sta"], stla, stlo))
                rec_count += 1

    with open(OUT_STA, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sta", "stla", "stlo"])
        for sta, la, lo in sorted(station_set):
            w.writerow([sta, f"{la:.4f}", f"{lo:.4f}"])

    print(f"tele events appended : {n_events}")
    print(f"tele arrivals        : {rec_count}")
    print(f"tele stations        : {len(station_set)}")
    print(f"wrote {OUT_FILE}")
    print(f"wrote {OUT_STA}")


if __name__ == "__main__":
    main()
