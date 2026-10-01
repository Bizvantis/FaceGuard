"""FaceGuard - face recognition, liveness detection and presence security for Windows.

Pipeline overview
-----------------
camera frame
  -> YuNet CNN face detector (bounding box + 5 landmarks)
  -> IoU tracker (stable identities across frames)
  -> landmark-based alignment to a 112x112 canonical face
  -> SFace CNN embedding (128-d, L2-normalised)
  -> open-set matcher (top-k cosine similarity vs. enrolled gallery)
  -> MediaPipe face mesh (478 landmarks + blendshapes) for liveness challenges
  -> presence guard (auto-lock / intruder capture / shoulder-surf alerts)
"""

import os as _os

# Quiet the native libraries (OpenCV DNN, MediaPipe/TFLite) before anything imports them.
_os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
_os.environ.setdefault("GLOG_minloglevel", "2")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

__version__ = "1.0.0"
APP_NAME = "FaceGuard"
