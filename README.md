# EAREDSinger

**Singing Voice Synthesis with Accompaniment-Referenced Expression Prediction via Denoising**

EAREDSinger is a research fork of [OpenVPI DiffSinger](https://github.com/openvpi/DiffSinger) that uses the musical accompaniment to guide singing expression. Given a score, lyrics, and a corresponding instrumental track, its variance model predicts pitch and editable expression curves with reference to the accompaniment. The acoustic model and vocoder remain those of OpenVPI DiffSinger.

The results below are preliminary objective measurements. Listening tests and ablation studies have not yet been completed.

## Method

1. The accompaniment waveform is converted to a frame-aligned mel spectrogram. The reported setup uses 44.1 kHz audio, a 512-sample hop, and 128 mel bins spanning 40 Hz to 16 kHz.
2. A single linear layer projects the accompaniment features into the 256-dimensional hidden space. Its output is added to the frame-level condition after the length regulator, before the pitch and multi-variance predictors. This adds 33,024 parameters to the variance model.
3. The predictors use that condition throughout their denoising steps to generate pitch and, in the reported experiment, breathiness, voicing, and tension curves. The linguistic encoder, duration predictor, acoustic model, and vocoder are unchanged by this accompaniment branch. Users can still edit or replace the predicted curves.

## Preliminary results

The report compares EAREDSinger with an otherwise comparable variance-model baseline without accompaniment. Both were evaluated at 44,000 training steps on the same held-out Opencpop data, split by song. The table shows mean R² ± sample standard deviation over three sampling seeds; higher is better.

| Predicted curve | Baseline R² | EAREDSinger R² |
| --- | ---: | ---: |
| Pitch | 0.9112 ± 0.0021 | **0.9189 ± 0.0005** |
| Breathiness | 0.4339 ± 0.0031 | **0.4594 ± 0.0015** |
| Voicing | 0.4409 ± 0.0097 | **0.5510 ± 0.0067** |
| Tension | 0.6292 ± 0.0048 | **0.6389 ± 0.0039** |

The seven held-out songs contain no training segments. R² was calculated over voiced, non-padding frames. For breathiness, voicing, and tension, both models received ground-truth pitch, so these numbers measure expression prediction given pitch. They do not establish end-to-end synthesis quality or listener preference. A formal listening test is still pending.

Demo: please refer to this [google slide](https://docs.google.com/presentation/d/16I35a1NNsat5lupPPBS8Nibf2UndPAedrA_HAYDCIMI/edit?usp=sharing).

## Getting started

For installation and the general preprocessing, training, and inference workflow, see [Getting Started](docs/GettingStarted.md). The [Best Practices](docs/BestPractices.md) and [Configuration Schemas](docs/ConfigurationSchemas.md) cover the underlying DiffSinger settings.

The repository includes an accompaniment-enabled example, [`config_variance_accompaniment.yaml`](config_variance_accompaniment.yaml). When `use_accompaniment: true`, variance preprocessing expects a matching `.wav` or `.flac` for each training item in the configured raw data directory's `accompaniments/` folder, using the same item name as the vocal recording. Prepare and align these tracks with the corresponding vocals before binarization.

Variance inference takes a DS file. If you have a full-length accompaniment WAV aligned with the DS file's time offsets, use [`slice_wav_by_ds.py`](scripts/slice_wav_by_ds.py) to create the corresponding WAV for each segment, then pass the output directory to a trained accompaniment-enabled experiment:

```bash
python scripts/slice_wav_by_ds.py path/to/accompaniment.wav my_song.ds --output-dir path/to/accompaniment_segments
python scripts/infer.py variance my_song.ds --exp my_experiment --accompaniment path/to/accompaniment_segments
```

The slicing script requires an uncompressed PCM WAV. It starts each slice at the DS segment's `offset` and uses `ph_dur` (or `note_dur` when `ph_dur` is absent) for its duration. It writes `0.wav`, `1.wav`, and so on in DS segment order, which is the naming expected by inference. See `python scripts/slice_wav_by_ds.py --help` and `python scripts/infer.py variance --help` for other options.

The accompaniment-conditioned branch currently supports inference through the Python command above. It has not been integrated into ONNX export or OpenUTAU; the upstream DiffSinger documentation for those deployment paths does not cover this branch.

## Upstream project and references

EAREDSinger builds on the [OpenVPI-maintained DiffSinger repository](https://github.com/openvpi/DiffSinger). Its general user documentation and architecture resources come from that project. The original research is [*DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism*](https://arxiv.org/abs/2105.02446), with the [original implementation](https://github.com/MoonInTheRiver/DiffSinger). This fork retains the upstream project's Apache 2.0 licensing and credits its contributors.

## Disclaimer

Do not use this repository to generate a person's voice without their consent, including the voices of public figures and celebrities. Such use may violate applicable rights or laws.

## License

This fork is licensed under the [Apache 2.0 License](LICENSE).
