"""Phase 4 3D reference environment.

A small, deterministic Python implementation of the Unreal research arena
used to develop and test the Python side of the Unreal bridge where Unreal
Engine itself is not available. It provides the same observations (rendered
third-person camera frames), the same actions and the same navigation
service over the same protocol (:mod:`chest_hunter.unreal_bridge`).

This package is *simulator side*: it owns ground truth. The visual agent
never imports it; it only sees what the bridge protocol delivers.
"""
