# Experiments behind the FaceGuard paper

Every number in `paper/faceguard.tex` comes from these scripts. Results are written to
`results/*.json`, and `make_figures.py` turns them into the paper's figures
(`paper/figures/*.pdf`), generated table rows and a macro file (`paper/generated/numbers.tex`)
that the paper `\input`s. No number in the paper is typed in by hand.

| Script | What it measures | Paper |
|---|---|---|
| `common.py` | Downloads LFW (scikit-learn, ~173 MB) and embeds all 13,233 images (3 pipeline variants) | - |
| `exp_lfw.py` | View-2 verification + ablations, FAR/FRR over 10^6 impostor pairs, identification, open-set DIR@FAR, classifiers | Tables II-IV, Figs. 5-7 |
| `exp_baselines.py` | Same-protocol ArcFace baselines (InsightFace `w600k_mbf`, `w600k_r50`), per-fold accuracies; `--latency` times the embedders | Table II |
| `exp_robustness.py` | Blur, brightness, resolution, JPEG, noise, mask, sunglasses (1,200 pairs + 1.2M impostor comparisons per condition) | Fig. 8, Sec. V-C |
| `exp_pose.py` | 5-landmark yaw vs MediaPipe 3-D pose on 2,998 photos; yaw jitter | Sec. III-C, V-F |
| `exp_parallax.py` | Yaw ratio vs homography parallax for heads, flat and bent prints; fits K and Kp on a calibration half (150 meshes) and evaluates on the disjoint half | Fig. 2, Sec. V-F |
| `exp_liveness.py` | Monte-Carlo evaluation of the challenge-response protocol on the evaluation meshes: ablations, conventional baselines, behavioural sensitivity | Tables VII-X, Fig. 9 |
| `exp_guard.py` | Presence guard on simulated webcam streams (owner scenario: 40 owners x 15 min; `--only-a` keeps earlier B-D results) | Table XI |
| `exp_guard_diag.py` | Failure analysis of the guard runs: classifies every trial's stranger, finds owner photos with a second person | Sec. V-H |
| `exp_mpii.py` | Recognition on laptop-webcam frames (MPIIFaceGaze, 15 people, cross-day pairs + enrollment protocol), SFace and ArcFace | Table V |
| `exp_demographics.py` | FMR/FNMR per demographic group on LFW (Kumar et al. attribute labels, within-group impostors, Wilson CIs) | Table VI |
| `exp_realworld.py` | Guided real-world test on the laptop webcam (production engine, log-only guard; no images stored) | Table XII |
| `exp_realworld_analysis.py` | Summarises results/realworld/*.jsonl (pilot v1 vs corrected protocol v2) | Sec. V-I |
| `exp_runtime.py` | Per-stage latency, CPU load (run on an otherwise idle machine) | Fig. 10, Sec. V-J |
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
..\.venv\Scripts\python.exe exp_demographics.py  # needs data/lfw_attributes.txt
..\.venv\Scripts\python.exe exp_realworld.py   # interactive, uses the webcam (close the FaceGuard app)
..\.venv\Scripts\python.exe exp_realworld_analysis.py
..\.venv\Scripts\python.exe exp_runtime.py     # close other programs first
..\.venv\Scripts\python.exe exp_baselines.py --latency   # idle machine
..\.venv\Scripts\python.exe make_figures.py
cd ..\paper && ..\tools\tectonic.exe faceguard.tex
```

The ArcFace models come from the InsightFace v0.7 release (`buffalo_l.zip`, `buffalo_sc.zip`;
non-commercial research use only). SHA-256: `w600k_r50.onnx` 4c06341c...619e43,
`w600k_mbf.onnx` 9cc6e4a7...319eb4f.

Hardware used for the paper: AMD Ryzen 7 7730U (8 cores / 16 threads), 16 GB RAM, Windows 11,
CPU only. The datasets and models are downloaded into `experiments/data/` (delete it to free ~700 MB).
