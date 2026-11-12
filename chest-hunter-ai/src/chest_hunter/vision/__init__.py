"""Visual perception (Phase 3): observations, preprocessing, detection, tracking.

Requires NumPy and OpenCV (``opencv-python-headless``). Perception modules
receive only rendered frames, their timestamp and the documented camera
geometry; they never read simulator state.

OpenCV is limited to one thread: the images are small, so multi-threading
added CPU overhead without reducing latency.
"""

import cv2 as _cv2

_cv2.setNumThreads(1)
