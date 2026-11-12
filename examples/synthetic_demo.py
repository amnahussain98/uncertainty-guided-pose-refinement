"""Small self-contained demo on a synthetic walking skeleton (no datasets or models needed).

Five "augmented crops" per frame are simulated: every joint has an error shared by the crops and a smaller
crop-specific error; during two occlusion episodes the left arm and the right leg become much noisier.
The spread of the five crops is therefore larger where the error is larger, as observed for real estimators.

Run:  python examples/synthetic_demo.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ugpr import SMPL_PARENTS, augmentation_spread, refine_person

OFFSETS = np.array([  # rest-pose offset of every SMPL joint from its parent, mm (x left, y up, z forward)
    [0, 0, 0], [70, -90, 0], [-70, -90, 0], [0, 110, 0], [0, -380, 0], [0, -380, 0], [0, 130, 0], [0, -400, 0],
    [0, -400, 0], [0, 60, 0], [0, -50, 120], [0, -50, 120], [0, 210, 0], [80, 120, 0], [-80, 120, 0], [0, 90, 50],
    [110, 30, 0], [-110, 30, 0], [0, -260, 0], [0, -260, 0], [0, -250, 0], [0, -250, 0], [0, -80, 0], [0, -80, 0]], float)


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def walking(T, fps=30.0):
    # forward kinematics with sinusoidal swings of the hips, knees, shoulders and elbows (1 step per second)
    t = np.arange(T) / fps
    swing = {1: (0.5, 0), 2: (0.5, np.pi), 4: (0.6, -0.8), 5: (0.6, np.pi - 0.8),
             16: (0.4, np.pi), 17: (0.4, 0), 18: (0.3, np.pi + 0.5), 19: (0.3, 0.5)}
    X = np.zeros((T, 24, 3))
    for k in range(T):
        R = [np.eye(3)] * 24
        for j in range(24):
            p = SMPL_PARENTS[j]
            if p < 0:
                continue
            amp, ph = swing.get(p, (0.0, 0.0))
            Rl = rot_x(amp * np.sin(2 * np.pi * t[k] + ph) + (0.4 if p in (4, 5) else 0.0))
            R[j] = R[p] @ Rl
            X[k, j] = X[k, p] + R[p] @ Rl @ OFFSETS[j]
    return X


def simulate_crops(gt, rng, n_aug=5):
    T, J = gt.shape[:2]
    s = np.full((T, J), 20.0)                      # typical per-joint error scale, mm
    s[60:100, [18, 20, 22]] = 120.0                # occlusion episode 1: left arm
    s[150:190, [5, 8, 11]] = 120.0                 # occlusion episode 2: right leg
    shared = rng.normal(0, 1, (T, J, 3)) * s[..., None]
    crops = gt[:, None] + shared[:, None] + rng.normal(0, 1, (T, n_aug, J, 3)) * 0.2 * s[:, None, :, None]
    return crops


def mpjpe(P, G):
    return float(np.linalg.norm((P - P[:, :1]) - (G - G[:, :1]), axis=-1).mean())


def accel(P, G):
    a = P[2:] - 2 * P[1:-1] + P[:-2]; b = G[2:] - 2 * G[1:-1] + G[:-2]
    return float(np.linalg.norm(a - b, axis=-1).mean())


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    gt = walking(240)
    crops = simulate_crops(gt, rng)
    poses, sigmas = augmentation_spread(crops)          # mean root-relative pose and per-joint spread
    gt_rr = gt - gt[:, :1]
    frames = np.arange(len(gt))
    print(f'{"method":18s} MPJPE (mm)  Accel (mm/frame^2)')
    print(f'{"per frame":18s} {mpjpe(poses, gt_rr):9.1f}  {accel(poses, gt_rr):9.1f}')
    for preset in ('online', 'offline'):
        est = refine_person(frames, poses, sigmas, preset=preset)
        R = np.stack([est[f] for f in frames])
        print(f'{preset:18s} {mpjpe(R, gt_rr):9.1f}  {accel(R, gt_rr):9.1f}')
