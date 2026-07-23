# DiffSinger variance visualization

This program extracts the same frame-level features used by DiffSinger's
variance binarizer and creates a browser-based interactive timeline with audio
playback.

From the repository root:

```powershell
python -m visualization path\to\vocal.wav
```

The generated HTML opens automatically. The WAV is embedded into the HTML by
default, so the single HTML file can be copied from a server to a local computer
without breaking playback. It contains:

- native audio playback with a synchronized playhead;
- mouse-wheel and button timeline zoom;
- drag-to-pan and click-to-seek interaction;
- vocal waveform, MIDI pitch, energy, voicing, breathiness, and tension tracks;
- accompaniment waveform, spectrogram, and short-time RMS loudness tracks;
- selectable Y-axis minimum/maximum controls with an automatic-range reset.

By default the program uses DiffSinger's checkpoint-free WORLD harmonic/noise
separator. To use the repository's Vocal Remover path:

```powershell
python -m visualization vocal.wav `
  --separator vr `
  --vr-checkpoint checkpoints\vr\model.pt
```

For a server-to-local workflow:

```bash
# Run this on the server.
python -m visualization /server/audio/vocal.wav \
  --accompaniment /server/audio/accompaniment.wav \
  --output /server/results/vocal-variance.html \
  --no-open

# Copy only vocal-variance.html to the local computer and open it normally.
```

Embedding increases the HTML size because each audio file is base64 encoded.
When `--accompaniment` is supplied, playback, pause, seeking, and playback speed
stay synchronized with the vocal, and the page includes an accompaniment volume
control. Use `--link-audio` only when the HTML will stay on the same machine as
both WAV files. Use `--no-open` on a headless server. Run
`python -m visualization --help` for all analysis parameters.
