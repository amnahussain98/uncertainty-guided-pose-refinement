"""Core of the refinement module, copied unchanged from the experiment notebook (Step 3 helpers).

Uncertainty from test-time augmentation, the constant-velocity Kalman filter / Rauch-Tung-Striebel smoother
with uncertainty-dependent measurement noise, and the person-specific bone-length stage.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Uncertainty from test-time augmentation
# ---------------------------------------------------------------------------
def rootrel_mean_and_sigma(pred_aug, root=0):
    # pred_aug: (A, J, 3) absolute joint positions (mm) from A augmented crops.
    # Returns the mean root-relative pose (J, 3) and the per-joint spread sigma (J,) in mm,
    # i.e. the root-mean-square per-axis deviation of the A root-relative predictions.
    rr = pred_aug - pred_aug[:, root:root + 1]
    mean = rr.mean(0)
    sigma = np.sqrt(((rr - mean) ** 2).sum(-1).mean(0) / 3.0)
    return mean, sigma


# ---------------------------------------------------------------------------
# Constant-velocity Kalman filter / Rauch-Tung-Striebel smoother, run independently
# for every joint coordinate (vectorized over D = joints x 3 dimensions).
# ---------------------------------------------------------------------------
def kalman_track(z, r, q, max_gap=15, smooth=False):
    # z: (T, D) measurements, NaN where missing.  r: (T, D) measurement variances.
    # q: process-noise intensity (mm^2 per frame^3).  Gaps longer than max_gap frames
    # start a new segment.  Returns (T, D) estimates (NaN where no estimate exists).
    T, D = z.shape
    out = np.full((T, D), np.nan)
    have = ~np.isnan(z).any(1)
    idx = np.nonzero(have)[0]
    if len(idx) == 0:
        return out
    # split into segments at long gaps
    segs, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a - 1 > max_gap:
            segs.append((start, a))
            start = b
    segs.append((start, idx[-1]))
    Q00, Q01, Q11 = q / 3.0, q / 2.0, q
    for s, e in segs:
        n = e - s + 1
        xp = np.zeros((n, 2, D)); Pp = np.zeros((n, 3, D))   # predicted (P00, P01, P11)
        xf = np.zeros((n, 2, D)); Pf = np.zeros((n, 3, D))   # filtered
        p = z[s].copy(); v = np.zeros(D)
        P00 = r[s].copy(); P01 = np.zeros(D); P11 = np.full(D, 1e4)
        for k in range(n):
            t = s + k
            if k > 0:  # predict
                p = p + v
                P00, P01, P11 = P00 + 2 * P01 + P11 + Q00, P01 + P11 + Q01, P11 + Q11
            xp[k, 0], xp[k, 1] = p, v
            Pp[k] = P00, P01, P11
            if have[t] and k > 0:  # update
                S = P00 + r[t]
                K0, K1 = P00 / S, P01 / S
                innov = z[t] - p
                p = p + K0 * innov
                v = v + K1 * innov
                P00, P01, P11 = (1 - K0) * P00, (1 - K0) * P01, P11 - K1 * P01
            xf[k, 0], xf[k, 1] = p, v
            Pf[k] = P00, P01, P11
        if smooth:
            ps, vs = xf[-1, 0].copy(), xf[-1, 1].copy()
            out[e] = ps
            for k in range(n - 2, -1, -1):
                f00, f01, f11 = Pf[k]
                a00, a01, a11 = Pp[k + 1]
                det = a00 * a11 - a01 * a01
                i00, i01, i11 = a11 / det, -a01 / det, a00 / det
                # C = Pf F^T Pp^-1 with F = [[1,1],[0,1]]  ->  Pf F^T = [[f00+f01, f01],[f01+f11, f11]]
                b00, b01, b10, b11 = f00 + f01, f01, f01 + f11, f11
                C00 = b00 * i00 + b01 * i01; C01 = b00 * i01 + b01 * i11
                C10 = b10 * i00 + b11 * i01; C11 = b10 * i01 + b11 * i11
                dp, dv = ps - xp[k + 1, 0], vs - xp[k + 1, 1]
                ps = xf[k, 0] + C00 * dp + C01 * dv
                vs = xf[k, 1] + C10 * dp + C11 * dv
                out[s + k] = ps
        else:
            out[s:e + 1] = xf[:, 0]
    return out


def refine_track(frames, poses, sigmas, mode, q, lam=1.0, smin=5.0, rfix=100.0, max_gap=15, smooth=False):
    # frames: sorted frame indices with a measurement; poses: (n, J, 3) root-relative;
    # sigmas: (n, J).  mode: 'fixed' (constant R = rfix) or 'uncertainty' (R = lam*max(sigma, smin)^2).
    # Returns dict frame -> refined pose (J, 3), including frames inside short gaps.
    if len(frames) == 0:
        return {}
    f0, f1 = frames[0], frames[-1]
    T, J = f1 - f0 + 1, poses.shape[1]
    z = np.full((T, J * 3), np.nan)
    r = np.full((T, J * 3), 1.0)
    for f, P, s in zip(frames, poses, sigmas):
        z[f - f0] = P.reshape(-1)
        if mode == 'fixed':
            r[f - f0] = rfix
        else:
            r[f - f0] = np.repeat(lam * np.maximum(s, smin) ** 2, 3)
    est = kalman_track(z, r, q, max_gap=max_gap, smooth=smooth)
    res = {}
    for k in range(T):
        if not np.isnan(est[k]).any():
            res[f0 + k] = est[k].reshape(J, 3)
    return res


# ---------------------------------------------------------------------------
# Skeletons: parent of every joint (-1 = root) and left/right bone pairs (by child joint)
# ---------------------------------------------------------------------------
SMPL_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])
SMPL_SYM = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23)]
MPI17_NAMES = ['htop', 'neck', 'rsho', 'relb', 'rwri', 'lsho', 'lelb', 'lwri', 'rhip', 'rkne', 'rank',
               'lhip', 'lkne', 'lank', 'pelv', 'spin', 'head']
MPI17_PARENTS = np.array([16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, -1, 14, 1])
MPI17_SYM = [(2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13)]
MPI17_ROOT = 14


def topo_order(parents):
    # Joints ordered so that every parent comes before its children.
    order, frontier = [], [int(np.where(parents < 0)[0][0])]
    while frontier:
        j = frontier.pop(0)
        order.append(j)
        frontier += [int(c) for c in np.where(parents == j)[0]]
    return order


# ---------------------------------------------------------------------------
# Bone-length consistency (new): a person's bone lengths do not change over time.
# Lengths are estimated from the confident (low-sigma) frames of a track and then used to
# correct the uncertain joints, blending by the joint's uncertainty.
# ---------------------------------------------------------------------------
def bone_lengths(P, parents):
    ch = np.where(parents >= 0)[0]
    L = np.full(P.shape[:-1], np.nan)
    L[..., ch] = np.linalg.norm(P[..., ch, :] - P[..., parents[ch], :], axis=-1)
    return L


def bone_weights(S, parents, s0):
    ch = np.where(parents >= 0)[0]
    W = np.zeros(S.shape)
    W[..., ch] = 1.0 / (S[..., ch] ** 2 + S[..., parents[ch]] ** 2 + s0 ** 2)
    return W


def weighted_median(x, w):
    # x, w: (n,) -> weighted median of x
    i = np.argsort(x)
    c = np.cumsum(w[i])
    k = int(np.searchsorted(c, 0.5 * c[-1]))
    return float(x[i][min(k, len(x) - 1)])


def bone_groups(parents, sym):
    # Groups of child joints whose bones share one length (left/right pairs if sym).
    ch = [int(j) for j in np.where(parents >= 0)[0]]
    groups, used = [], set()
    if sym:
        for a, b in sym:
            groups.append([a, b]); used |= {a, b}
    groups += [[j] for j in ch if j not in used]
    return groups


def estimate_bones(poses, sigmas, parents, sym=None, s0=5.0, how='median', online=False):
    # poses (n, J, 3), sigmas (n, J).  Offline: one length per bone, shape (J,), NaN at the root.
    # Online: causal estimate for every frame from that frame and the frames before it, shape (n, J).
    L = bone_lengths(poses, parents)
    W = bone_weights(sigmas, parents, s0)
    n, J = L.shape
    out = np.full((n, J) if online else (J,), np.nan)
    for g in bone_groups(parents, sym):
        if online:
            num = np.cumsum((W[:, g] * L[:, g]).sum(1))
            den = np.cumsum(W[:, g].sum(1))
            for j in g:
                out[:, j] = num / den
        else:
            x, w = L[:, g].reshape(-1), W[:, g].reshape(-1)
            v = weighted_median(x, w) if how == 'median' else float((w * x).sum() / w.sum())
            for j in g:
                out[j] = v
    return out


def apply_bones(X, Lhat, parents, order, S, kappa=1.0, smin=3.0, iters=10):
    # Uncertainty-weighted projection of each pose onto the track's bone lengths.
    # X (n, J, 3) root-relative poses; Lhat (J,) or (n, J) target lengths; S (n, J) joint uncertainties.
    # Each bone constraint |Y_j - Y_p| = L is enforced by moving both ends along the bone, the more
    # uncertain end moving more (inverse mass = max(sigma, smin)^2; the root stays fixed). A few
    # Gauss-Seidel sweeps approximate the weighted least-squares projection. kappa in [0, 1] blends
    # between the input (0) and the projected pose (1).
    n, J = X.shape[:2]
    Lf = np.broadcast_to(Lhat, (n, J)) if np.ndim(Lhat) == 1 else Lhat
    w = np.maximum(S, smin) ** 2
    w[:, parents < 0] = 0.0
    Y = X.copy()
    bones = [(j, parents[j]) for j in order if parents[j] >= 0]
    for _ in range(iters):
        for j, p in bones:
            v = Y[:, j] - Y[:, p]
            ln = np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)
            ws = w[:, j:j + 1] + w[:, p:p + 1]
            lam = (ln - Lf[:, j:j + 1]) / np.maximum(ws, 1e-9)
            d = v / ln
            Y[:, j] -= w[:, j:j + 1] * lam * d
            Y[:, p] += w[:, p:p + 1] * lam * d
    return X + kappa * (Y - X)


# ---------------------------------------------------------------------------
# Full refinement: optional Kalman filter / RTS smoother (helpers_v3) + optional bone stage
# cfg = dict(mode='fixed'|'uncertainty'|None, q, lam, smin, rfix, smooth,
#            bones=None | dict(kappa, bsmin, s0, how, sym, order='after'|'before'))
# ---------------------------------------------------------------------------
def _bone_stage(X, S, poses, sigmas, frames, fs, bones, smooth, parents, order, sym_pairs):
    sym = sym_pairs if bones.get('sym') else None
    if smooth:
        Lh = estimate_bones(poses, sigmas, parents, sym, bones.get('s0', 5.0), bones.get('how', 'median'))
    else:  # causal: lengths from measured frames up to each frame (gap frames use the last estimate)
        Lc = estimate_bones(poses, sigmas, parents, sym, bones.get('s0', 5.0), online=True)
        pos = np.searchsorted(frames, fs, side='right') - 1
        Lh = Lc[np.maximum(pos, 0)]
    return apply_bones(X, Lh, parents, order, S, bones.get('kappa', 1.0), bones.get('bsmin', 3.0))


def refine_v4(frames, poses, sigmas, cfg, parents, order, sym_pairs, max_gap=15):
    if len(frames) == 0:
        return {}
    frames = np.asarray(frames)
    bones = cfg.get('bones')
    smooth = cfg.get('smooth', False)
    P = poses
    if bones is not None and bones.get('order', 'after') == 'before':
        P = _bone_stage(poses, sigmas, poses, sigmas, frames, frames, bones, smooth, parents, order, sym_pairs)
    if cfg.get('mode') is None:
        est = {int(f): p for f, p in zip(frames, P)}
    else:
        kc = {k: cfg[k] for k in ('mode', 'q', 'lam', 'smin', 'rfix', 'smooth') if k in cfg}
        est = refine_track(frames, P, sigmas, max_gap=max_gap, **kc)
    if bones is None or bones.get('order', 'after') != 'after' or not est:
        return est
    fs = np.array(sorted(est))
    X = np.stack([est[f] for f in fs])
    meas = {int(f): k for k, f in enumerate(frames)}
    med = np.median(sigmas, axis=0)
    S = np.stack([sigmas[meas[f]] if f in meas else med for f in fs])   # gap frames: typical sigma
    Y = _bone_stage(X, S, poses, sigmas, frames, fs, bones, smooth, parents, order, sym_pairs)
    return {int(f): y for f, y in zip(fs, Y)}
