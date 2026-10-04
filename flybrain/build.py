"""Build a simulatable network from the MaleCNS v1.0 connectome.

The recipe follows the one described for ornata/fly (Fly64): all superclass-annotated
neurons, synapse-count weights, sign from the predicted neurotransmitter, and each
neuron's inputs normalized to sum to 1. Also records where each photoreceptor sits in the eye so a
1-D "scene" (opponent to the left/right, near/far) can be projected onto it.

Output: <DATA>/brain.npz  (weights as CSR, neuron groups, eye azimuths)

    flybrain build [--data DIR]      (needs pip install "flybrain[build]")
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
from scipy import sparse

from .data import DATA

BUCKET = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
SOURCES = {
    "body-annotations-male-cns-v1.0-minconf-0.5.feather": f"{BUCKET}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather": f"{BUCKET}/body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather": f"{BUCKET}/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "optic-columns.xlsx": "https://raw.githubusercontent.com/flyconnectome/2025malecns/"
                          "67767d2233657983993ff6c2be48e836a935863c/supplemental_data/optic-column-type-assignments-v1.0.xlsx",
}


def download_all(raw: Path) -> None:
    """Fetch the MaleCNS v1.0 tables (~1.1 GB) into `raw` unless already present."""
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        target = raw / name
        if target.exists():
            continue
        partial = target.with_suffix(target.suffix + ".part")
        print(f"downloading {name}")

        def progress(blocks, block_size, total):
            if total > 0:
                print(f"\r  {min(blocks * block_size, total) / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="")

        urllib.request.urlretrieve(url, partial, progress)
        print()
        partial.replace(target)
INHIBITORY = "gaba|glutamate|histamine"

# Descending neurons (brain -> body commands) used as motor read-outs.
# Types follow the literature: DNa02 steering, DNp01 giant-fiber escape,
# DNg100 forward walking (as in fly64), MDN = moonwalker (backward walking).
# punch/kick have no fly equivalent; these picks are arbitrary: DNg11 (a
# descending neuron with no known fighting role) and pIP10 (the male
# courtship-song command neuron). Nothing we tested drives either of them.
MOTOR_TYPES = {
    "forward": ["DNg100"],
    "steer": ["DNa02"],
    "escape": ["DNp01"],
    "backward": ["MDN"],
    "punch": ["DNg11"],
    "kick": ["pIP10"],
}


COLUMN_NAME = re.compile(r"ME_(?P<side>[LR])_col_(?P<h1>\d+)_(?P<h2>\d+)")
COLUMN_CELLS = ("L1", "R7", "R8")    # neurons the spreadsheet assigns to each medulla column


def optic_columns(path: Path) -> dict[int, tuple[str, int, int]]:
    """body id -> (eye side, h1, h2): the hexagonal medulla column each L1 / R7 / R8 neuron
    belongs to, from the MaleCNS optic-column spreadsheet (one sheet per optic lobe)."""
    import pandas as pd

    placed = {}
    for sheet in pd.read_excel(path, sheet_name=None, dtype=str).values():
        if "column" not in sheet.columns:
            continue                                  # the READ ME sheet
        where = sheet["column"].str.extract(COLUMN_NAME)
        for cell in COLUMN_CELLS:
            if cell not in sheet.columns:
                continue
            body = pd.to_numeric(sheet[cell], errors="coerce")
            ok = where["side"].notna() & (body > 0)
            for b, side, h1, h2 in zip(body[ok], where["side"][ok], where["h1"][ok], where["h2"][ok]):
                placed[int(b)] = (side, int(h1), int(h2))
    return placed


def build(data: Path | str = DATA) -> None:
    """Download MaleCNS v1.0 into <data>/raw and write weights.npz, brain.npz and brain.json to `data`."""
    data = Path(data)
    raw = data / "raw"
    download_all(raw)
    ann = feather.read_table(raw / "body-annotations-male-cns-v1.0-minconf-0.5.feather").to_pandas()
    nt = feather.read_table(raw / "body-neurotransmitters-male-cns-v1.0.feather",
                            columns=["body", "consensus_nt"]).to_pandas()

    ann = ann.loc[ann["superclass"].notna() & ann["superclass"].ne("")]
    ann = ann.drop_duplicates("bodyId").sort_values("bodyId").set_index("bodyId")
    ids = ann.index.to_numpy(np.int64)
    n = len(ids)
    print(f"{n:,} neurons")

    labels = nt.drop_duplicates("body").set_index("body").reindex(ids)["consensus_nt"]
    sign = np.where(labels.fillna("unclear").str.lower().str.contains(INHIBITORY), -1.0, 1.0).astype(np.float32)

    edges = feather.read_table(raw /"connectome-weights-male-cns-v1.0-minconf-0.5.feather",
                               columns=["body_pre", "body_post", "weight"], memory_map=True)
    pre_parts, post_parts, w_parts = [], [], []
    for i, batch in enumerate(edges.to_batches(max_chunksize=4_000_000), 1):
        pre_id = batch.column(0).to_numpy(zero_copy_only=False)
        post_id = batch.column(1).to_numpy(zero_copy_only=False)
        pre = np.minimum(np.searchsorted(ids, pre_id), n - 1)
        post = np.minimum(np.searchsorted(ids, post_id), n - 1)
        ok = (ids[pre] == pre_id) & (ids[post] == post_id)
        pre_parts.append(pre[ok].astype(np.int32))
        post_parts.append(post[ok].astype(np.int32))
        w_parts.append(batch.column(2).to_numpy(zero_copy_only=False)[ok].astype(np.float32))
        print(f"\r  scanned {min(i * 4_000_000, edges.num_rows):,} / {edges.num_rows:,} edge rows", end="")
    print()
    del edges
    pre, post, w = (np.concatenate(p) for p in (pre_parts, post_parts, w_parts))
    del pre_parts, post_parts, w_parts
    w *= sign[pre]
    incoming = np.bincount(post, weights=np.abs(w), minlength=n).astype(np.float32)
    w /= np.maximum(incoming[post], 1.0)
    W = sparse.csr_matrix((w, (post, pre)), shape=(n, n), dtype=np.float32)
    print(f"{W.nnz:,} connections")

    cell_type = ann["flywireType"].combine_first(ann["type"]).fillna("").astype(str)
    side = ann["somaSide"].fillna(ann["rootSide"]).fillna("").astype(str).str.upper()
    instance = ann["instance"].fillna("").astype(str).str.upper()

    # Soma (or soma-tract) position of each neuron, in EM voxels; NaN if unknown.
    def xyz(v):
        return np.asarray(v, np.float32) if isinstance(v, (list, np.ndarray)) and len(v) == 3 else None

    soma = ann["somaLocation"].map(xyz).combine_first(ann["tosomaLocation"].map(xyz))
    positions = np.stack([np.full(3, np.nan, np.float32) if v is None else v for v in soma])

    def pick(types, want_side=None):
        mask = cell_type.isin(types)
        if want_side:
            mask &= (side == want_side) | instance.str.contains(f"_{want_side}")
        return np.flatnonzero(mask.to_numpy()).astype(np.int32)

    groups = {}
    for name, types in MOTOR_TYPES.items():
        groups[f"{name}_L"] = pick(types, "L")
        groups[f"{name}_R"] = pick(types, "R")

    # Photoreceptors and their azimuth in the eye (-1 = far left ... +1 = far right).
    visual = pick(["R1-6", "R7", "R8"])
    columns = optic_columns(raw / "optic-columns.xlsx")
    known = np.array([i for i, b in enumerate(ids) if int(b) in columns], np.int32)
    to_known = abs(W[known][:, visual]).tocsc()  # R1-6 -> strongest column-assigned partner
    h1_max = max(h1 for _, h1, _ in columns.values())
    azimuth = np.full(len(visual), np.nan, np.float32)
    eye = np.array([side.iat[v] for v in visual])
    for k, neuron in enumerate(visual):
        loc = columns.get(int(ids[neuron]))
        if loc is None:
            a, b = to_known.indptr[k:k + 2]
            if b > a:
                loc = columns[int(ids[known[to_known.indices[a + np.argmax(to_known.data[a:b])]]])]
        if loc is not None:
            eye_side, h1, _ = loc
            eye[k] = eye_side
            # h1 runs front->back within an eye (approximate, like fly64).
            frac = (h1 - 1) / max(h1_max - 1, 1)
            azimuth[k] = -(0.06 + 0.94 * frac) if eye_side == "L" else (0.06 + 0.94 * frac)
    missing = np.isnan(azimuth)
    azimuth[missing] = np.where(eye[missing] == "L", -0.5, 0.5)  # unplaced: mid-eye guess
    print(f"{len(visual):,} photoreceptors ({missing.sum()} without a column estimate)")

    for name, idx in groups.items():
        print(f"  {name:11s} {len(idx):3d} neurons  {sorted(set(cell_type.iloc[idx]))}")
    if any(len(g) == 0 for g in groups.values()):
        raise RuntimeError("a motor group resolved to zero neurons; check MOTOR_TYPES")

    sparse.save_npz(data / "weights.npz", W, compressed=False)
    np.savez(data / "brain.npz", ids=ids, visual=visual, azimuth=azimuth,
             cell_type=cell_type.to_numpy().astype(str), side=side.to_numpy().astype(str),
             positions=positions, superclass=ann["superclass"].to_numpy().astype(str),
             **{f"group_{k}": v for k, v in groups.items()})
    (data / "brain.json").write_text(json.dumps(
        {"neurons": n, "connections": int(W.nnz), "photoreceptors": int(len(visual)),
         "groups": {k: int(len(v)) for k, v in groups.items()}}, indent=2))
    print(f"saved to {data}")


if __name__ == "__main__":
    build()
