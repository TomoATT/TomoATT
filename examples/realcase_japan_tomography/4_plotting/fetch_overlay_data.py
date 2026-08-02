#!/usr/bin/env python3
"""
fetch_overlay_data.py

Download the open geoscience datasets drawn as overlays on the Japan
tomography figures (all public, citable products):

  overlay_data/PB2002_boundaries.json      global plate boundaries
       (fraxen/tectonicplates digitisation of Bird 2003, PB2002)
  overlay_data/gem_active_faults.geojson   GEM Global Active Faults (Styron &
       Pagani 2020), filtered to the Japan box at plot time
  overlay_data/gvp_volcanoes.csv           Smithsonian GVP Holocene volcano
       catalogue (TidyTuesday 2020-05-12 mirror of volcano.si.edu database)
  overlay_data/<zone>_slab1.0_clip.grd     USGS Slab1.0 (Hayes et al. 2012)
       clipped slab-top grids for the Japan subduction system:
       kur (Kuril-Japan Pacific slab), izu (Izu-Bonin Pacific slab),
       ryu (Ryukyu), phi (Philippine Sea slab under SW Japan); grids taken
       from the official usgs/slab2 repository library.

Coastlines come from cartopy's Natural Earth download at plot time.

Usage: python fetch_overlay_data.py
"""

import sys
from pathlib import Path

import requests

OUT = Path(__file__).parent.resolve() / "overlay_data"
OUT.mkdir(exist_ok=True)

ASSETS = {
    "PB2002_boundaries.json":
        "https://raw.githubusercontent.com/fraxen/tectonicplates/master/"
        "GeoJSON/PB2002_boundaries.json",
    "gem_active_faults.geojson":
        "https://raw.githubusercontent.com/GEMScienceTools/"
        "gem-global-active-faults/master/geojson/"
        "gem_active_faults_harmonized.geojson",
    "gvp_volcanoes.csv":
        "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/"
        "data/2020/2020-05-12/volcano.csv",
    "kur_slab1.0_clip.grd":
        "https://raw.githubusercontent.com/usgs/slab2/master/slab2code/library/"
        "slab1grids/kur_slab1.0_clip.grd",
    "izu_slab1.0_clip.grd":
        "https://raw.githubusercontent.com/usgs/slab2/master/slab2code/library/"
        "slab1grids/izu_slab1.0_clip.grd",
    "phi_slab1.0_clip.grd":
        "https://raw.githubusercontent.com/usgs/slab2/master/slab2code/library/"
        "slab1grids/notused/phi_slab1.0_clip.grd",
    "ryu_slab1.0_clip.grd":
        "https://raw.githubusercontent.com/usgs/slab2/master/slab2code/library/"
        "slab1grids/notused/ryu_slab1.0_clip.grd",
}

CHUNK = 1 << 20


def fetch(name, url):
    dst = OUT / name
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  exists   : {name} ({dst.stat().st_size/1e6:.1f} MB)")
        return
    print(f"  download : {name} <- {url}")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
    print(f"           : {dst.stat().st_size/1e6:.2f} MB")


def main():
    for name, url in ASSETS.items():
        try:
            fetch(name, url)
        except requests.RequestException as e:
            print(f"  FAILED   : {name}: {e}", file=sys.stderr)
    print("done.")


if __name__ == "__main__":
    main()
