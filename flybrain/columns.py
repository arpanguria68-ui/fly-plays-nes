"""Where each motion-detector cell (T4, T5) looks in the eye, and which way it prefers.

The MaleCNS optic-column table places L1, R7 and R8 in their medulla columns (hexagonal
coordinates h1, h2). Everything else here is read off the wiring:

* column: each neuron on the ON/OFF motion pathway (L2-L5, Mi1, Tm3, Mi4, Mi9, Tm1-4,
  Tm9, C3, T4, T5) takes the contact-weighted mean column of its three strongest already
  placed partners, in rounds outward from L1 (T4 <- Mi1 <- L1 is two synapses).
* preferred direction: a T4 cell takes its slow, delayed input (Mi9) from one side of its
  column and its fast, inhibitory input (Mi4) from the other. Motion from the Mi9 side
  toward the Mi4 side is its preferred direction (Takemura et al. 2017; Borst 2018).
  Measured over each subtype this is 92-97% consistent, and T4a/T4b and T4c/T4d come out
  opposite. The subtype means then define each eye's axes: T4a (front-to-back) points
  backward, T4c (upward) points up. T5's delayed partner CT1 has no column, so a T5 takes
  the direction of the T4 subtype sharing its lobula plate layer (T5a = T4a, ...).

eye_map() computes this once (a few seconds) and caches it as <data>/columns.npz:
cells, eye ("L"/"R"), front (0 = back of the eye ... 1 = front), up (0 = bottom ... 1 =
top), pref (unit vector (front, up) the cell prefers motion toward).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
from scipy import sparse

from .data import ensure_data

PATHWAY = ("L2", "L3", "L4", "L5", "Mi1", "Tm3", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9", "C3")
MOTION = tuple(f"{k}{d}" for k in ("T4", "T5") for d in "abcd")
COLUMNS_URL = ("https://raw.githubusercontent.com/flyconnectome/2025malecns/"
               "67767d2233657983993ff6c2be48e836a935863c/supplemental_data/optic-column-type-assignments-v1.0.xlsx")


def _place(W: sparse.csr_matrix, ct: np.ndarray, ids: np.ndarray, columns: dict) -> tuple[np.ndarray, np.ndarray]:
    n = len(ids)
    xy = np.full((n, 2), np.nan, np.float32)
    eye = np.full(n, "", "<U1")
    for body, (side, h1, h2) in columns.items():
        i = np.searchsorted(ids, body)
        if i < n and ids[i] == body:
            xy[i] = (h1 - 0.5 * h2, np.sqrt(3) / 2 * h2)
            eye[i] = side
    A = (abs(W) + abs(W).T).tocsr()                # contact strength, either direction
    cand = np.flatnonzero(np.isin(ct, PATHWAY + MOTION))
    while True:
        placed = ~np.isnan(xy[:, 0])
        new = 0
        for i in cand[~placed[cand]]:
            a, b = A.indptr[i:i + 2]
            nb, w = A.indices[a:b], A.data[a:b]
            k = placed[nb]
            if not k.any():
                continue
            nb, w = nb[k], w[k]
            top = np.argsort(-w)[:3]
            top = top[eye[nb[top]] == eye[nb[top[0]]]]     # partners in the same eye
            xy[i] = (xy[nb[top]] * w[top, None]).sum(0) / w[top].sum()
            eye[i] = eye[nb[top[0]]]
            new += 1
        if not new:
            return xy, eye


def _offsets(W: sparse.csr_matrix, ct: np.ndarray, xy: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Mi9 -> Mi4 input offset of each T4 cell (unit vector, hex plane); NaN if missing."""
    out = np.full((len(cells), 2), np.nan, np.float32)
    for k, i in enumerate(cells):
        a, b = W.indptr[i:i + 2]
        pre, w = W.indices[a:b], np.abs(W.data[a:b])
        ends = []
        for t in ("Mi9", "Mi4"):
            m = (ct[pre] == t) & ~np.isnan(xy[pre, 0])
            ends.append((xy[pre[m]] * w[m, None]).sum(0) / w[m].sum() if m.any() else None)
        if ends[0] is not None and ends[1] is not None:
            v = ends[1] - ends[0]
            if np.hypot(*v) > 1e-3:
                out[k] = v / np.hypot(*v)
    return out


def build_eye_map(data: Path, spreadsheet: Path) -> dict:
    from .build import optic_columns                   # needs pandas + openpyxl
    meta = np.load(data / "brain.npz")
    W = sparse.load_npz(data / "weights.npz").tocsr()   # rows = postsynaptic
    ct, ids = meta["cell_type"].astype(str), meta["ids"]
    xy, eye = _place(W, ct, ids, optic_columns(spreadsheet))
    cells = np.flatnonzero(np.isin(ct, MOTION) & (eye != ""))
    t4 = cells[np.char.startswith(ct[cells], "T4")]
    own = dict(zip(t4, _offsets(W, ct, xy, t4)))
    # each eye's axes, from the subtype means: T4a prefers front-to-back, T4c upward
    axes = {}
    for side in "LR":
        mean = {d: np.nanmean([own[i] for i in t4 if ct[i] == f"T4{d}" and eye[i] == side], axis=0) for d in "abcd"}
        back = (mean["a"] - mean["b"]) / 2
        up = (mean["c"] - mean["d"]) / 2
        axes[side] = (np.linalg.inv(np.column_stack([-back, up])), mean)   # hex plane -> (front, up)
    front_up = np.zeros((len(cells), 2), np.float32)
    pref = np.zeros((len(cells), 2), np.float32)
    for k, i in enumerate(cells):
        to_fu, mean = axes[eye[i]]
        front_up[k] = to_fu @ xy[i]
        v = own.get(i)
        if v is None or np.isnan(v[0]):
            v = mean[ct[i][2]]                               # the T4 twin's subtype direction
        p = to_fu @ v
        pref[k] = p / max(np.hypot(*p), 1e-6)
    for side in "LR":                                        # 0..1 within each eye
        m = eye[cells] == side
        lo, hi = front_up[m].min(0), front_up[m].max(0)
        front_up[m] = (front_up[m] - lo) / (hi - lo)
    return {"cells": cells.astype(np.int32), "eye": eye[cells], "front": front_up[:, 0],
            "up": front_up[:, 1], "pref": pref, "types": ct[cells]}


def eye_map(data: Path | str | None = None) -> dict:
    """The T4/T5 eye map (see module docstring), computed once and cached in <data>/columns.npz."""
    data = ensure_data(data)
    cache = data / "columns.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    sheet = data / "raw" / "optic-columns.xlsx"
    if not sheet.exists():
        sheet.parent.mkdir(parents=True, exist_ok=True)
        print("downloading the MaleCNS optic-column table (110 kB)...", flush=True)
        urllib.request.urlretrieve(COLUMNS_URL, sheet)
    print("mapping T4/T5 cells onto the eye (once)...", flush=True)
    out = build_eye_map(data, sheet)
    np.savez(cache, **out)
    return out
