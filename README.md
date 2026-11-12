# Uncertainty-Guided Temporal Refinement With Bone-Length Consistency

Code for the paper

> A. Hussain, F. Zafar and Z. Gillani, "Uncertainty-Guided Temporal Refinement With Bone-Length Consistency for
> Occlusion-Robust 3D Human Pose Estimation," submitted to *IEEE Access*, 2026.

A 3D pose estimator is run on a few augmented crops of each person (original, mirrored, zoomed out, shifted,
darker). Where the crops disagree, a joint is unreliable. This disagreement sets the measurement noise of a Kalman
filter (online) or Rauch–Tung–Striebel smoother (offline), and it weights a correction toward the person's own
bone lengths, which are estimated from the confident frames of the track. No extra network is trained. On 3DPW with
ResNet-50 MeTRAbs, offline refinement lowers MPJPE from 68.2 to 66.3 mm and acceleration error from 20.9 to
4.9 mm/frame².

## Contents

| Path | What it is |
|---|---|
| `ugpr/` | The refinement module (NumPy only): augmentation spread, Kalman/RTS with uncertainty-dependent noise, bone-length stage |
| `examples/synthetic_demo.py` | A small demo on a synthetic walking skeleton (no data needed) |
| `experiments/MeTRAbs_3DPW_Experiments.ipynb` | The Google Colab notebook that produces every number and figure of the paper |
| `tests/` | Basic checks of the module |

## Using the module

```bash
pip install numpy scipy
python examples/synthetic_demo.py
```

```python
import numpy as np
from ugpr import augmentation_spread, refine_person

# pred_aug: (T, A, 24, 3) SMPL-joint predictions in mm from A augmented crops of one tracked person,
# mapped back to the original image (mirrored crops un-mirrored, left/right joints swapped back).
poses, sigmas = augmentation_spread(pred_aug)                # mean root-relative pose, per-joint spread
frames = np.arange(len(poses))                               # frame index of every detection of this person
refined = refine_person(frames, poses, sigmas, preset='offline')   # or preset='online' (causal)
# refined: {frame: (24, 3) pose}; frames inside gaps of up to 15 frames are filled
```

Other skeletons work by passing `parents` (parent of each joint, −1 for the root) and `sym` (left/right joint
pairs that share one bone length); MPI-INF-3DHP/MuPoTS 17-joint definitions are included (`MPI17_PARENTS`,
`MPI17_SYM`). The presets are the settings chosen on the 3DPW validation split for ResNet-50 MeTRAbs; for another
estimator they should be re-selected on its own validation data (see Part I of the notebook).

## Reproducing the paper

1. Open `experiments/MeTRAbs_3DPW_Experiments.ipynb` in Google Colab and choose a T4 GPU runtime.
2. Read the data terms in the first cell. Results are written to a folder `3DPW_results_v4` in your Google Drive;
   datasets are downloaded into the Colab session only.
3. Run all cells once with `QUICK_TEST = True` (a few minutes), then with `QUICK_TEST = False`. If Colab
   disconnects, run all cells again: finished pieces are read from Drive and skipped.

| Paper item | Notebook part |
|---|---|
| Tables 2a, 2b, 3; Fig. 3 | Parts A, C, E1–E4 |
| Table 2c (MuPoTS-3D, default camera) | Part G |
| Table 4 (shuffled-uncertainty controls), Table 5 (bone-stage ablation) | Part F |
| Fig. 4 (sensitivity, number of crops) | Parts E6, F |
| Table 6 (tracker) | Parts E5, F |
| Table 7 (EfficientNetV2-M) | Parts D, E7 |
| Table 8 (HMR 2.0) | Part I |
| Table 9, Fig. 5 (per-frame occlusion and truncation) | Parts B, E7 |
| Tables 10, 11 (occlusion episodes) | Parts B, E7, G |
| Fig. 6 (uncertainty quintiles) | Parts E4, E7 |
| Figs. 7–9 (examples) | Parts H, H2 |
| Gap filling, lag, bone lengths (Section V-H), speed | Parts E, F, G |

## Data and models

Nothing from the datasets or the pretrained models is included in this repository. The notebook downloads them
from their official sources, and their own terms apply:
3DPW (non-commercial research; cite von Marcard et al., ECCV 2018), MuPoTS-3D (Mehta et al., 3DV 2018),
Pascal VOC 2012 (Flickr terms of use), MeTRAbs (Sárándi et al.), SmoothNet (Zeng et al., ECCV 2022),
HMR 2.0 / 4D-Humans (Goel et al., ICCV 2023) and the SMPL body model (Loper et al., 2015; registration at
https://smpl.is.tue.mpg.de, needed for Part I only).

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## Licence

The code in this repository is released under the MIT licence (see `LICENSE`). The datasets and models it
downloads keep their own licences.
