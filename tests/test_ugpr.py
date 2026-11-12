import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ugpr import (SMPL_PARENTS, SMPL_SYM, kalman_track, augmentation_spread, refine_person, estimate_bones,
                  apply_bones, topo_order)
from ugpr.core import bone_lengths


def test_kalman_constant_signal():
    z = np.full((50, 3), 7.0)
    for smooth in (False, True):
        out = kalman_track(z, np.ones_like(z), q=10.0, smooth=smooth)
        assert np.allclose(out, 7.0)


def test_gap_is_filled_and_long_gap_is_not():
    z = np.arange(60, dtype=float)[:, None].repeat(3, 1)
    z[20:25] = np.nan          # short gap: filled
    z[35:55] = np.nan          # long gap (> 15 frames): new segment, not filled
    out = kalman_track(z, np.ones_like(z), q=1.0, max_gap=15, smooth=True)
    assert not np.isnan(out[20:25]).any() and np.isnan(out[35:55]).all()
    assert np.allclose(out[22], 22.0, atol=0.5)


def test_bone_projection_restores_lengths():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 200, (10, 24, 3)); X[:, 0] = 0
    S = np.full((10, 24), 10.0)
    L = estimate_bones(X, S, SMPL_PARENTS, SMPL_SYM, how='mean')
    Y = apply_bones(X, L, SMPL_PARENTS, topo_order(SMPL_PARENTS), S, kappa=1.0, iters=50)
    ch = SMPL_PARENTS >= 0
    err0 = np.nanmean(np.abs(bone_lengths(X, SMPL_PARENTS)[:, ch] - L[ch]))
    err1 = np.nanmean(np.abs(bone_lengths(Y, SMPL_PARENTS)[:, ch] - L[ch]))
    assert err1 < 0.05 * err0


def test_refine_person_runs():
    rng = np.random.default_rng(1)
    crops = rng.normal(0, 30, (40, 5, 24, 3))
    poses, sigmas = augmentation_spread(crops)
    assert poses.shape == (40, 24, 3) and sigmas.shape == (40, 24)
    for preset in ('online', 'offline', 'online_no_bones', 'offline_no_bones'):
        est = refine_person(np.arange(40), poses, sigmas, preset=preset)
        assert len(est) == 40 and est[0].shape == (24, 3)


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f(); print(name, 'ok')
