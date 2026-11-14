"""Phase 5: demonstrations, imitation learning, model export and policy selection.

Training (``behavior_cloning``) is offline and separate from the control loop; the control
loop only runs inference (``learned_policy``) through the safety layer in
``policy_selector``. NumPy is the only hard dependency; ONNX / ONNX Runtime are optional
(export and faster CPU inference).
"""
