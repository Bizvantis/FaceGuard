# Experiments behind the FaceGuard paper

Every number in `paper/faceguard.tex` comes from these scripts. Results are written to
`results/*.json`, and `make_figures.py` turns them into the paper's figures
(`paper/figures/*.pdf`), generated table rows and a macro file (`paper/generated/numbers.tex`)
that the paper `\input`s. No number in the paper is typed in by hand.

| Script | What it measures | Paper |
|---|---|---|
| `common.py` | Downloads LFW (scikit-learn, ~173 MB) and embeds all 13,233 images (3 pipeline variants) | - |
| `exp_lfw.py` | View-2 verification + ablations, FAR/FRR over 10^6 impostor pairs, identification, open-set DIR@FAR, classifiers | Tables IV-VI, Figs. 5-7 |
| `exp_baselines.py` | Same-protocol ArcFace baselines (InsightFace `w600k_mbf`, `w600k_r50`), per-fold accuracies; `--latency` times the embedders | Table IV |
| `exp_bootstrap.py` | Identity-level (subject) bootstrap intervals for the LFW operating points and the MPIIFaceGaze protocols | Table V, Sec. V-D |
| `exp_robustness.py` | Blur, brightness, resolution, JPEG, noise, mask, sunglasses (1,200 pairs + 1.2M impostor comparisons per condition) | Fig. 8, Sec. V-C |
| `exp_pose.py` | 5-landmark yaw vs MediaPipe 3-D pose on 2,998 photos; yaw jitter | Sec. III-C, V-F |
| `exp_parallax.py` | Yaw ratio vs homography parallax for heads, flat and bent prints; fits K and Kp on a calibration half (150 meshes) and evaluates on the disjoint half | Fig. 2, Sec. V-F |
| `exp_parallax_real.py` | Real vs simulated parallax: real-world turn frames, landmark-noise sweep, lens distortion of the 15 calibrated MPIIFaceGaze webcams | Fig. 9, Sec. V-F |
| `exp_liveness.py` | Monte-Carlo evaluation of the challenge-response protocol on the evaluation meshes: ablations, conventional baselines, behavioural sensitivity | Tables IX-XII, Fig. 10 |
| `exp_liveness_extra.py` | Five-seed repetitions with a face-level bootstrap, adaptive replay attackers, threshold and landmark-noise sensitivity | Sec. V-G |
| `exp_guard.py` | Presence guard on simulated webcam streams (owner scenario: 40 owners x 15 min; `--only-a` keeps earlier B-D results) | Table XIII |
| `exp_guard_diag.py` | Failure analysis of the guard runs: classifies every trial's stranger, finds owner photos with a second person | Sec. V-H |
| `exp_mpii.py` | Recognition on laptop-webcam frames (MPIIFaceGaze, 15 people, cross-day pairs + enrollment protocol), SFace and ArcFace | Table VII |
| `exp_adaptation.py` | Production template-update rule over weeks of MPIIFaceGaze frames: genuine scores, impostor acceptance and poisoning | Sec. V-D |
| `exp_demographics.py` | FMR/FNMR per demographic group on LFW (Kumar et al. attribute labels, within-group impostors, Wilson CIs) | earlier version |
| `exp_demographics_backbones.py` | Demographic differentials with balanced identity-level bootstrap intervals for SFace and both ArcFace models at a matched FMR | Table VIII |
| `exp_realworld.py` | Guided real-world test on the laptop webcam (production engine, log-only guard; no images stored) | Table XIV |
| `exp_realworld_analysis.py` | Summarises results/realworld/*.jsonl: round 1 (earlier protocol), round 2 (corrected protocol) and all phone attacks pooled | Sec. V-I, Table XIV |
| `exp_physical.py` | Physical study (interactive, webcam): bona fide sessions of several users (volunteers enrolled in memory only), attacks with phone/screen photos, prompt-following and looped video replays and prints, the turn-only gate randomised per trial (none / landmark / tracked-feature), live scores of four PAD baselines, static holds, continuous guard sessions with scripted events; numbers only | Sec. V-I |
| `exp_physical_analysis.py` | `--freeze`: freezes the tracked-feature gate from the owner's bona fide turns; then attack success, paired gate analysis, cue separation, ISO/IEC 30107-3 APCER/BPCER/ACER of PAD baselines and FaceGuard, guard false/missed locks per hour and per event | Sec. V-I |
| `pad_models.py` | ONNX wrappers of the PAD baselines (MiniFASNet V2+V1SE, anti-spoof-mn3, He et al. CVPRW 2024 ResNet-50 and Swin-V2-B) with their authors' preprocessing | Sec. V-I |
| `pad_convert.py` | Converts the two CVPRW 2024 PyTorch checkpoints to ONNX (separate PyTorch env); `--macs` counts their cost | Sec. V-I |
| `exp_runtime.py` | Per-stage latency, CPU load (run on an otherwise idle machine) | Fig. 11, Sec. V-J |
| `exp_olivetti.py` | AT&T/Olivetti benchmark (not used in the current paper) | - |
| `make_figures.py` | Figures, `numbers.tex`, generated tables, Wilson intervals and significance tests | all |

## Reproduce

```bash
cd experiments
..\.venv\Scripts\python.exe common.py          # LFW download + embeddings (~10 min)
..\.venv\Scripts\python.exe exp_lfw.py
..\.venv\Scripts\python.exe exp_baselines.py   # needs data/insightface/{buffalo_l,buffalo_sc} and onnxruntime
..\.venv\Scripts\python.exe exp_robustness.py  # ~25 min
..\.venv\Scripts\python.exe exp_parallax.py    # before exp_pose/exp_liveness (writes face_shapes.npz + split)
..\.venv\Scripts\python.exe exp_pose.py
..\.venv\Scripts\python.exe exp_liveness.py    # ~30 min on 8 processes
..\.venv\Scripts\python.exe exp_guard.py       # ~1.5 h
..\.venv\Scripts\python.exe exp_guard_diag.py  # ~5 min, after exp_guard.py
..\.venv\Scripts\python.exe exp_mpii.py        # needs data/mpiifacegaze (MPIIFaceGaze.zip, 942 MB)
..\.venv\Scripts\python.exe exp_adaptation.py  # same data, ~3 min
..\.venv\Scripts\python.exe exp_bootstrap.py   # identity-level intervals (caches results/mpii_sface.npz)
..\.venv\Scripts\python.exe exp_demographics_backbones.py  # ArcFace on all of LFW (~25 min, cached)
..\.venv\Scripts\python.exe exp_demographics.py  # needs data/lfw_attributes.txt
..\.venv\Scripts\python.exe exp_realworld.py   # interactive, uses the webcam (close the FaceGuard app)
..\.venv\Scripts\python.exe exp_realworld_analysis.py
..\.venv\Scripts\python.exe exp_liveness_extra.py   # ~30 min; after exp_parallax.py
..\.venv\Scripts\python.exe exp_parallax_real.py    # after exp_realworld_analysis.py (uses the recorded traces)
..\.venv\Scripts\python.exe exp_physical.py    # interactive: owner calibration first (see the docstring)
..\.venv\Scripts\python.exe exp_physical_analysis.py --freeze   # once, before any attack session
..\.venv\Scripts\python.exe exp_physical.py    # attacks, volunteers, guard sessions
..\.venv\Scripts\python.exe exp_physical_analysis.py
..\.venv\Scripts\python.exe exp_runtime.py     # close other programs first
..\.venv\Scripts\python.exe exp_baselines.py --latency   # idle machine
..\.venv\Scripts\python.exe make_figures.py
cd ..\paper && ..\tools\tectonic.exe faceguard.tex
```

PAD baselines (downloaded into `data/pad/`): `MiniFASNetV2.onnx`, `MiniFASNetV1SE.onnx` from the releases of
github.com/yakhyo/face-anti-spoofing (Apache-2.0); `anti-spoof-mn3.onnx` from the OpenVINO Open Model Zoo
(SHA-384 6de45349...ece5d1d4); `full_resnet50.pth` and `face_swin_v2_base.pth` from
github.com/Xianhua-He/cvpr2024-face-anti-spoofing-challenge (MIT), converted with `pad_convert.py`.

The ArcFace models come from the InsightFace v0.7 release (`buffalo_l.zip`, `buffalo_sc.zip`;
non-commercial research use only). SHA-256: `w600k_r50.onnx` 4c06341c...619e43,
`w600k_mbf.onnx` 9cc6e4a7...319eb4f.

Hardware used for the paper: AMD Ryzen 7 7730U (8 cores / 16 threads), 16 GB RAM, Windows 11,
CPU only. The datasets and models are downloaded into `experiments/data/` (delete it to free ~700 MB).
