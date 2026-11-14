# Training artefacts (Phase 5)

* `configs/bc_default.json` - behaviour-cloning configuration used for the committed model.
* `datasets/synthetic_expert_v1.stats.json` - statistics, validation report and the
  episode-level train/validation/test split of the demonstration dataset. The dataset
  itself (about 370 MB of PNG frames) is not committed. Recreate it with:

  ```bash
  python -m chest_hunter record-demo --controller scripted --map research_arena_3d \
      --episodes 6 --seed 1000 --output training/datasets/synthetic_expert_v1 --workers 3
  for i in $(seq -f "%03g" 0 11); do
    python -m chest_hunter record-demo --controller scripted --map proc_train_$i \
        --episodes 2 --seed 1000 --output training/datasets/synthetic_expert_v1 --workers 2
  done
  ```

  These are **synthetic** demonstrations (the deterministic Phase 4 controller), not
  human play. Human demonstrations: `record-demo --controller human` (none recorded yet).
* `checkpoints/bc_synthetic_v1/` - the trained policy: `policy.npz` (NumPy weights),
  `policy.onnx` (FP32), `policy_int8.onnx` (INT8), `policy.json` (architecture,
  normalisation, dataset statistics, validation/test metrics), `history.json`,
  `export.json` (export checks and CPU latency).
* `experience.db` (not committed) - sessions and observations of the demonstration runs;
  the spawn estimator and route memory learn from it.
