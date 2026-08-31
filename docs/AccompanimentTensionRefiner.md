# Accompaniment-conditioned tension refiner

The frozen variance model predicts base tension without accompaniment. The
second-stage refiner reads the full-resolution full-song accompaniment mel.
A local convolutional encoder preserves transients and broadband level before
learned 16x temporal reduction. A non-causal Transformer then predicts a coarse
additive `accompaniment_delta` with access to the entire song.

## Prepare the song-level dataset

Set `base_variance_ckpt` in `config_tension_refiner.yaml`, then run:

```powershell
python scripts/prepare_tension_refiner.py `
  --config config_tension_refiner.yaml
```

The current architecture requires rebuilding any dataset made by the former
mean-pooled-mel implementation. Pass `--overwrite` to do that intentionally.
Training-set per-bin and broadband-level normalization statistics are computed
during preparation and stored in both the dataset metadata and checkpoint.
Preparation uses actual base-model inference without ground-truth pitch or
variance inputs. Segment names such as `2001_001` are placed on the original
timeline using the `句子` tier of the corresponding full-song TextGrid.

## Train

```powershell
python scripts/train.py `
  --config config_tension_refiner.yaml `
  --exp_name opencpop_tension_refiner
```

TensorBoard reports both `base_tension_r2` and `refined_tension_r2`.

## Export a refined DS

The input DS must already contain `tension` and `tension_timestep` for every
segment. These are treated as the frozen base model's curves.

```powershell
python scripts/infer_tension_refiner.py `
  --config config_tension_refiner.yaml `
  --accompaniment C:/path/to/song.wav `
  --ds C:/path/to/base.ds `
  --refiner-ckpt checkpoints/opencpop_tension_refiner/model_ckpt_steps_10000.ckpt `
  --output C:/path/to/refined.ds `
  --diagnostics C:/path/to/refined_tension.npz
```

The input checkpoint must use the current Transformer architecture; checkpoints
from the earlier TCN refiner are not shape-compatible. The output DS preserves
all other fields and each segment's original tension timestep and point count.
`--diagnostics` is optional; its NPZ contains the
full-song `base_tension`, `accompaniment_delta`, `tension`, `curve_mask`, and
`timestep` for inspection.
