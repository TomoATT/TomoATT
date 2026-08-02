#!/usr/bin/env python3
"""
make_warmstart_model.py

Convert a TomoATT in-process model (out_data_grid.h5, group 'model',
datasets vel_inv_XXXX / xi_inv_XXXX / eta_inv_XXXX) or a final_model.h5
into a valid init_model file (root datasets vel/xi/eta) so that a run
that was killed by the 48/72 h wall clock limit can be restarted from
its last completed inversion iteration.

Usage:
  python make_warmstart_model.py \
      --in  OUTPUT_FILES_joint_tele_ani/out_data_sim_group_0.h5 \
      --out restart_model_step12.h5 [--step 12]

If --step is omitted, the largest available step index is used.
Then in the input params: init_model_path -> restart_model_step12.h5,
and the restarting run covers the remaining iterations
(model_update.max_iterations: max_iter - step).
"""

import argparse
import re

import h5py


def pick(fg, prefix, step):
    keys = [k for k in fg.keys() if k.startswith(prefix)]
    if not keys:
        return None, None
    numbered = []
    for k in keys:
        m = re.search(r"(\d+)$", k)
        if m:
            numbered.append((int(m.group(1)), k))
    numbered.sort()
    if step is not None:
        for n, k in numbered:
            if n == step:
                return n, k
    n, k = numbered[-1]
    return n, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="fin", required=True)
    ap.add_argument("--out", dest="fout", required=True)
    ap.add_argument("--step", type=int, default=None)
    args = ap.parse_args()

    with h5py.File(args.fin, "r") as f:
        grp = f["model"] if "model" in f else f
        chosen = {}
        step_used = None
        for name, prefix in (("vel", "vel"), ("xi", "xi"), ("eta", "eta")):
            if name in grp:                     # final_model.h5 case
                chosen[name] = grp[name][:]
                continue
            n, key = pick(grp, f"{prefix}_inv_", args.step)
            if key is None and prefix == "vel":
                # fall back to un-numbered vel
                key = "vel" if "vel" in grp else None
                n = None
            if key is None:
                print(f"warning: no dataset for {prefix}; writing zeros later")
                chosen[name] = None
            else:
                chosen[name] = grp[key][:]
                if step_used is None:
                    step_used = n
    base = chosen["vel"]
    with h5py.File(args.fout, "w") as o:
        for name in ("vel", "xi", "eta"):
            arr = chosen[name]
            if arr is None:
                arr = 0.0 * base
            o.create_dataset(name, data=arr)
    print(f"wrote {args.fout} from step {step_used} (vel/xi/eta)")


if __name__ == "__main__":
    main()
