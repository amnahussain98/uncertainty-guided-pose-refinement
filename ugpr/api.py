"""Simple interface: refine one person's track from augmented 3D pose predictions."""
import copy
import numpy as np

from .core import (rootrel_mean_and_sigma, refine_v4, topo_order, SMPL_PARENTS, SMPL_SYM)

# Settings selected on the 3DPW validation split for ResNet-50 MeTRAbs (Table 2a of the paper).
# q: process noise; lam, smin: measurement variance lam * max(sigma, smin)^2; bones: bone-length stage.
PRESETS = {
    'offline': dict(mode='uncertainty', q=1000.0, lam=30.0, smin=10.0, smooth=True,
                    bones=dict(kappa=1.0, bsmin=3.0, s0=5.0, how='mean', sym=True, order='before')),
    'online': dict(mode='uncertainty', q=3000.0, lam=30.0, smin=3.0, smooth=False,
                   bones=dict(kappa=1.0, bsmin=3.0, s0=5.0, how='mean', sym=True, order='after')),
    # the same filters without the bone stage
    'offline_no_bones': dict(mode='uncertainty', q=1000.0, lam=30.0, smin=10.0, smooth=True),
    'online_no_bones': dict(mode='uncertainty', q=3000.0, lam=30.0, smin=3.0, smooth=False),
}


def augmentation_spread(pred_aug, root=0):
    """pred_aug: (T, A, J, 3) predictions of A augmented crops per frame (mm, aligned to the original image).
    Returns the mean root-relative poses (T, J, 3) and the per-joint spread sigma (T, J) in mm."""
    out = [rootrel_mean_and_sigma(np.asarray(p, np.float64), root) for p in pred_aug]
    return np.stack([m for m, _ in out]), np.stack([s for _, s in out])


def refine_person(frames, poses, sigmas, preset='offline', cfg=None, parents=SMPL_PARENTS, sym=SMPL_SYM, max_gap=15):
    """Refine one tracked person.

    frames : (n,) increasing integer frame indices at which the person was detected
    poses  : (n, J, 3) root-relative 3D poses in mm (e.g. the mean of the augmented predictions)
    sigmas : (n, J) per-joint augmentation spread in mm (see augmentation_spread)
    preset : 'offline' (RTS smoother, uses future frames), 'online' (causal Kalman filter), or the
             '*_no_bones' variants; or pass a full cfg dict instead.
    parents, sym : skeleton (parent of each joint, -1 for the root) and left/right pairs that share a length.
    max_gap: frames without detection that are filled; longer gaps start a new segment.

    Returns a dict {frame: (J, 3) refined pose}, including filled frames inside short gaps.
    """
    cfg = copy.deepcopy(PRESETS[preset] if cfg is None else cfg)
    parents = np.asarray(parents)
    return refine_v4(np.asarray(frames), np.asarray(poses, np.float64), np.asarray(sigmas, np.float64), cfg,
                     parents, topo_order(parents), sym, max_gap=max_gap)
