#!/usr/bin/env python3
"""Stage-1: real LoRA pack vs independent FP64 per-LoRA reference for all targets."""
import json, struct, math, os, sys
from dataclasses import dataclass, field
import numpy as np

OIL = "/path/to/loras/OiledSkin_Zit_Turbo_V1.safetensors"
LAND = "/path/to/loras/LandidzuStyle_ZIT.safetensors"

def load_sf(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        data_off = 8 + n
        meta = hdr.pop("__metadata__", {})
        tensors = {}
        for name, info in hdr.items():
            dtype = info["dtype"]
            shape = tuple(info["shape"])
            off0, off1 = info["data_offsets"]
            f.seek(data_off + off0)
            raw = f.read(off1 - off0)
            if dtype == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
                # bf16 -> f32
                arr = np.empty(shape, dtype=np.float32)
                arr.view(np.uint32)[:] = u16.astype(np.uint32) << 16
            elif dtype == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
            elif dtype == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            else:
                raise ValueError(dtype)
            tensors[name] = arr
    return tensors, meta

def f16_exact(x):
    return np.float16(x).astype(np.float32)

def ilog2_floor(m):
    if m == 0:
        return -999
    f, e = math.frexp(m)
    return e - 1

@dataclass
class Stats:
    values: int = 0
    inexact_values: int = 0
    max_abs_err: float = 0.0
    shifted_vectors: int = 0
    split_vectors: int = 0

@dataclass
class VecPart:
    k: int
    h: np.ndarray  # float32 values that are exact f16

def split_vec(v, max_parts, st: Stats):
    # v: 1d float32
    L = v.shape[0]
    res = v.astype(np.float64).copy()
    m = float(np.max(np.abs(res))) if L else 0.0
    st.values += L
    parts = []
    if m == 0:
        return parts
    def exact_at(k):
        t = np.ldexp(res, k).astype(np.float32)
        if np.any(np.abs(t) > 65504.0):
            return False
        roundtrip = f16_exact(t)
        return bool(np.all(roundtrip == t))
    ref_m = 0.0
    for p in range(max(1, max_parts)):
        mr = float(np.max(np.abs(res)))
        if mr == 0:
            break
        e = ilog2_floor(mr)
        kmax = 14 - e
        if p == 0:
            k0 = min(0, kmax)
            kcap = max(k0, min(kmax, -e))
        else:
            k0 = min(kmax, ilog2_floor(ref_m) - e)
            kcap = k0
        k = kcap
        exact = False
        for kk in range(k0, kcap + 1):
            if exact_at(kk):
                k = kk
                exact = True
                break
        if p == 0:
            ref_m = math.ldexp(mr, k)
        t = np.ldexp(res, k).astype(np.float32)
        h = f16_exact(t)
        res = res - np.ldexp(h.astype(np.float64), -k)
        parts.append(VecPart(k=k, h=h))
        if exact:
            break
    if any(vp.k != 0 for vp in parts):
        st.shifted_vectors += 1
    if len(parts) > 1:
        st.split_vectors += 1
    err = np.abs(res)
    nz = err != 0
    st.inexact_values += int(np.sum(nz))
    if nz.any():
        st.max_abs_err = max(st.max_abs_err, float(err.max()))
    return parts

@dataclass
class Combined:
    K: int
    N: int
    R: int
    Rp: int
    A: np.ndarray  # [Rp, K] f32 holding f16 values
    B: np.ndarray  # [N, Rp]
    s: np.ndarray  # [Rp]
    gain: int
    st: Stats = field(default_factory=Stats)

def combine(srcs, K, N, max_split=3):
    # srcs: list of dicts with down[r,K], up[n,r] wait: down (r,K) as rows? 
    # C: down[j*K+k], up[i*r+j]
    st = Stats()
    terms = []  # (src_idx, a_part, b_part)
    a_parts_all = []
    b_parts_all = []
    for si, ps in enumerate(srcs):
        r = ps["r"]
        down = ps["down"]  # shape (r, K) float32
        up = ps["up"]      # shape (n, r) float32
        assert down.shape == (r, K)
        assert up.shape[1] == r
        for j in range(r):
            ap = split_vec(down[j], max_split, st)
            bp = split_vec(up[:, j], max_split, st)
            a_parts_all.append(ap)
            b_parts_all.append(bp)
            for a in ap:
                for b in bp:
                    terms.append((si, a, b))
    max_scale = max(abs(ps["scale"]) for ps in srcs) if srcs else 0.0
    gain = max(0, min(10, -ilog2_floor(max_scale))) if max_scale > 0 else 0
    R = len(terms)
    Rp = max(32, (R + 31) // 32 * 32)
    A = np.zeros((Rp, K), dtype=np.float32)
    B = np.zeros((N, Rp), dtype=np.float32)
    s = np.zeros(Rp, dtype=np.float32)
    for t, (si, a, b) in enumerate(terms):
        ps = srcs[si]
        A[t] = a.h
        n = ps["n"]
        row_off = ps["row_off"]
        B[row_off:row_off + n, t] = b.h
        s[t] = np.float32(math.ldexp(float(ps["scale"]), -(a.k + b.k)))
    return Combined(K=K, N=N, R=R, Rp=Rp, A=A, B=B, s=s, gain=gain, st=st)

def reference_delta(srcs, x):
    # x: (M, K), return (M, N)
    M, K = x.shape
    N = max(ps["row_off"] + ps["n"] for ps in srcs)
    out = np.zeros((M, N), dtype=np.float64)
    for ps in srcs:
        # y = down @ x.T -> (r, M); down is (r,K)
        y = ps["down"].astype(np.float64) @ x.T  # (r, M)
        # up (n,r) @ y -> (n, M)
        d = (ps["up"].astype(np.float64) @ y) * float(ps["scale"])
        out[:, ps["row_off"]:ps["row_off"] + ps["n"]] += d.T
    return out

def packed_delta(c: Combined, x):
    M, K = x.shape
    # y[t] = s[t] * (A[t] @ x[m])
    # A is (Rp, K)
    y = (c.A.astype(np.float64) @ x.T)  # (Rp, M)
    y = y * c.s.astype(np.float64)[:, None]
    # B (N, Rp) @ y -> (N, M)
    return (c.B.astype(np.float64) @ y).T

def check_packed(srcs, c, M=4, seed=1):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1, 1, size=(M, c.K)).astype(np.float32)
    ref = reference_delta(srcs, x)
    got = packed_delta(c, x)
    em = np.max(np.abs(got - ref))
    rm = np.max(np.abs(ref))
    return (em / rm) if rm > 0 else em

def targets_from_lora(tensors):
    """Map diffusion_model.*.lora_A/B to (base_key -> A,B)."""
    outs = {}
    for name, arr in tensors.items():
        if not name.endswith(".lora_A.weight"):
            continue
        base = name[: -len(".lora_A.weight")]
        bname = base + ".lora_B.weight"
        if bname not in tensors:
            continue
        A = arr  # [r, K] diffusers
        B = tensors[bname]  # [N, r]
        r, K = A.shape
        N, r2 = B.shape
        assert r == r2
        # ggml layout: down [K,r] as (r,K) rows in our combine; up [r,n] as (n,r)
        outs[base] = {
            "down": A.copy(),          # (r, K) — matches down[j*K+k] if stored row-major j,k
            "up": B.copy(),            # (N, r)
            "K": K, "N": N, "r": r,
        }
    return outs

def run_case(name, oil_m, land_m, oil_t, land_t):
    keys = sorted(set(oil_t) | set(land_t))
    assert len(keys) == 180, len(keys)
    max_rel = 0.0
    worst = None
    gains = {}
    shifted = split = inexact = 0
    fail = 0
    for i, key in enumerate(keys):
        srcs = []
        K = N = None
        for mult, table, tag in [(oil_m, oil_t, "oil"), (land_m, land_t, "land")]:
            if key not in table:
                continue
            t = table[key]
            K, N, r = t["K"], t["N"], t["r"]
            srcs.append({
                "down": t["down"], "up": t["up"], "r": r, "n": N, "row_off": 0,
                "scale": float(mult),  # no alpha in files → sv=1
            })
        c = combine(srcs, K, N, 3)
        rel = check_packed(srcs, c, M=4, seed=(i * 2654435761) & 0xFFFFFFFF)
        gains[key] = c.gain
        shifted += c.st.shifted_vectors
        split += c.st.split_vectors
        inexact += c.st.inexact_values
        if rel > max_rel:
            max_rel = rel
            worst = (key, rel, c.gain, c.R, c.Rp)
        if rel > 1e-4 or c.st.inexact_values:
            fail += 1
            print(f"  FAIL {key} rel={rel:.3g} inexact={c.st.inexact_values} gain={c.gain}")
    w2_gains = {k: g for k, g in gains.items() if "feed_forward.w2" in k}
    print(f"CASE {name}: targets={len(keys)} max_rel={max_rel:.3g} worst={worst}")
    print(f"  shifted={shifted} split={split} inexact={inexact} fails={fail}")
    print(f"  gain histogram: { {g: list(gains.values()).count(g) for g in sorted(set(gains.values()))} }")
    print(f"  w2 gains ({len(w2_gains)}): { {g: list(w2_gains.values()).count(g) for g in sorted(set(w2_gains.values()))} }")
    return max_rel, fail, gains

def main():
    print("Loading LoRAs...")
    oil, _ = load_sf(OIL)
    land, _ = load_sf(LAND)
    oil_t = targets_from_lora(oil)
    land_t = targets_from_lora(land)
    print(f"OiledSkin targets={len(oil_t)} dtype=BF16")
    print(f"Landidzu targets={len(land_t)} dtype=F16")
    assert set(oil_t) == set(land_t)
    # User cases
    r1 = run_case("OiledSkin x0.7, Landidzu x1.0 (BAD phone)", 0.7, 1.0, oil_t, land_t)
    r2 = run_case("Landidzu x0.9, OiledSkin x0.7 (GOOD phone)", 0.7, 0.9, oil_t, land_t)
    # Also strength 2.0
    r3 = run_case("OiledSkin x0.7, Landidzu x2.0", 0.7, 2.0, oil_t, land_t)
    print("STAGE1_SUMMARY", {
        "bad_max_rel": r1[0], "bad_fails": r1[1],
        "good_max_rel": r2[0], "good_fails": r2[1],
        "x2_max_rel": r3[0], "x2_fails": r3[1],
        "bad_w2_gain": next(v for k,v in r1[2].items() if "w2" in k),
        "good_w2_gain": next(v for k,v in r2[2].items() if "w2" in k),
        "x2_w2_gain": next(v for k,v in r3[2].items() if "w2" in k),
    })

if __name__ == "__main__":
    main()
