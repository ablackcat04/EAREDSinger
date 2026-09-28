"""
Evaluate variance-model checkpoints on the FULL held-out (valid) split.

Why this script exists
----------------------
VarianceTask._validation_step only updates the R^2 metrics for items with
data_idx < num_valid_plots (the metrics are tied to plotting), so the R^2 shown
in TensorBoard covers only the first few segments of the first test song.
This script runs inference on every valid segment instead.

Evaluation condition
--------------------
Same as validation: the variance predictor receives the GROUND-TRUTH pitch
(run_model passes sample['pitch']), so breathiness/voicing/tension R^2 measure
"variance given pitch". Pitch R^2 is the pitch predictor's own output.
R^2 is pooled over all voiced, non-padding frames (identical to
modules.metrics.RawCurveR2Score), overall and per song.

Usage (run from the repo root)
------------------------------
python scripts/eval_variance_r2.py \
    --ckpt checkpoints/opencpop_normalized_no_accompaniment/model_ckpt_steps_52000.ckpt \
    --ckpt checkpoints/opencpop_normalized_accompaniment/model_ckpt_steps_42000.ckpt \
    --label baseline --label proposed \
    --seeds 0 1 2 --out eval_r2.json

Both models are evaluated on the SAME binary data (by default the binary_data_dir
of the last checkpoint, i.e. the accompaniment dataset), so they see identical
segments. Override with --data_dir.
"""
import argparse
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch

root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from utils.hparams import set_hparams, hparams  # noqa: E402


# hparams that determine how the GT curves in the binary data were extracted.
# If a model was trained on data binarized with different values, evaluating it
# on another model's binary data would compare it against targets it never saw.
BINARIZE_KEYS = [
    'audio_sample_rate', 'hop_size', 'win_size', 'fft_size', 'f0_min', 'f0_max',
    'pe', 'pe_ckpt', 'hnsep', 'hnsep_ckpt', 'midi_smooth_width',
    'breathiness_smooth_width', 'voicing_smooth_width', 'tension_smooth_width',
    'use_glide_embed', 'glide_types',
]


def check_model_against_data(label, own_cfg, ref_cfg, eval_names):
    ok = True
    # (a) every evaluation segment must be held out from THIS model's training set
    prefixes = own_cfg['_test_prefixes']
    leaked = [n for n in eval_names if not any(n.startswith(p) for p in prefixes)]
    if leaked:
        ok = False
        print(f'| WARNING [{label}] {len(leaked)}/{len(eval_names)} evaluation segments are NOT in its '
              f'test_prefixes, i.e. this model was trained on them (e.g. {leaked[:5]}).')
    # (b) GT curves must have been extracted the same way as in this model's own data
    diffs = {k: (own_cfg.get(k), ref_cfg.get(k)) for k in BINARIZE_KEYS if own_cfg.get(k) != ref_cfg.get(k)}
    if diffs:
        ok = False
        print(f'| WARNING [{label}] binarization settings differ from the evaluation data: {diffs}')
    if ok:
        print(f'| [{label}] OK: all evaluation segments held out; binarization settings match.')
    return ok


class R2Acc:
    """Pooled R^2 accumulator, same formula as RawCurveR2Score."""

    def __init__(self):
        self.s = 0.0    # sum of targets
        self.s2 = 0.0   # sum of squared targets
        self.rss = 0.0  # residual sum of squares
        self.n = 0

    def update(self, pred, target):
        pred = pred.double()
        target = target.double()
        self.s += target.sum().item()
        self.s2 += (target * target).sum().item()
        self.rss += ((target - pred) ** 2).sum().item()
        self.n += target.numel()

    def compute(self):
        if self.n == 0:
            return float('nan')
        return 1.0 - self.rss / (self.s2 - self.s ** 2 / self.n)


def to_device(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def load_task(ckpt_path: pathlib.Path, data_dir, device):
    cfg = ckpt_path.parent / 'config.yaml'
    assert cfg.exists(), f'config.yaml not found next to {ckpt_path}'
    set_hparams(config=str(cfg), print_hparams=False)
    hparams['work_dir'] = str(ckpt_path.parent)
    hparams['infer'] = False  # only toggles per-step tqdm bars in the samplers
    own_data_dir = hparams['binary_data_dir']
    if data_dir is not None:
        hparams['binary_data_dir'] = data_dir

    from training.variance_task import VarianceTask, VarianceDataset
    import utils

    task = VarianceTask()
    utils.load_ckpt(task.model, str(ckpt_path), prefix_in_ckpt='model', strict=True, device='cpu')
    task.model.to(device).eval()
    dataset = VarianceDataset('valid')
    own_cfg = {k: hparams.get(k) for k in BINARIZE_KEYS}
    own_cfg['_test_prefixes'] = [p for d in hparams.get('datasets', []) for p in d.get('test_prefixes', [])]
    return task, dataset, own_data_dir, own_cfg


@torch.no_grad()
def evaluate(task, dataset, device, seed, first_k):
    torch.manual_seed(seed)
    np.random.seed(seed)
    names = dataset.metadata['names']
    targets = []
    if task.predict_pitch:
        targets.append('pitch')
    targets += list(task.variance_prediction_list)

    overall = {t: R2Acc() for t in targets}
    first = {t: R2Acc() for t in targets}
    per_song = defaultdict(lambda: {t: R2Acc() for t in targets})

    for i in range(len(dataset)):
        batch = to_device(dataset.collater([dataset[i]]), device)
        _, pitch_pred, var_pred = task.run_model(batch, infer=True)
        mask = (batch['mel2ph'] > 0) & ~batch['uv']
        song = names[i].split('_')[0]

        preds = {}
        if task.predict_pitch:
            preds['pitch'] = batch['base_pitch'] + pitch_pred
        for name in task.variance_prediction_list:
            preds[name] = var_pred[name]

        for t in targets:
            p = preds[t][mask]
            g = batch[t][mask]
            overall[t].update(p, g)
            per_song[song][t].update(p, g)
            if i < first_k:
                first[t].update(p, g)

    return {
        'overall': {t: overall[t].compute() for t in targets},
        'first_k': {t: first[t].compute() for t in targets},
        'per_song': {s: {t: acc[t].compute() for t in targets} for s, acc in sorted(per_song.items())},
        'n_segments': len(dataset),
        'names': list(names),
    }


def mean_std(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', required=True, help='checkpoint path (repeat for each model)')
    ap.add_argument('--label', action='append', help='label for each --ckpt, same order')
    ap.add_argument('--data_dir', default=None, help='binary data dir used for ALL models '
                                                     '(default: binary_data_dir of the last --ckpt)')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--first_k', type=int, default=3,
                    help='also report R^2 on the first k segments, to reproduce the old TensorBoard numbers')
    ap.add_argument('--out', default='eval_r2.json')
    args = ap.parse_args()

    ckpts = [pathlib.Path(c) for c in args.ckpt]
    labels = args.label or [c.parent.name for c in ckpts]
    assert len(labels) == len(ckpts), '--label count must match --ckpt count'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    import yaml
    with open(ckpts[-1].parent / 'config.yaml', encoding='utf-8') as f:
        ref_hp = yaml.safe_load(f)  # config whose binarization produced the evaluation data
    ref_cfg = {k: ref_hp.get(k) for k in BINARIZE_KEYS}
    data_dir = args.data_dir if args.data_dir is not None else ref_hp['binary_data_dir']
    print(f'| evaluating all models on: {data_dir}  (device={device})')

    results = {}
    ref_names = None
    for ckpt, label in zip(ckpts, labels):
        task, dataset, own_dir, own_cfg = load_task(ckpt, data_dir, device)
        if pathlib.Path(own_dir).resolve() != pathlib.Path(data_dir).resolve():
            print(f'| NOTE [{label}] was trained with binary_data_dir={own_dir}; evaluated on {data_dir}.')
        check_model_against_data(label, own_cfg, ref_cfg, list(dataset.metadata['names']))
        runs = [evaluate(task, dataset, device, s, args.first_k) for s in args.seeds]
        if ref_names is None:
            ref_names = runs[0]['names']
        assert runs[0]['names'] == ref_names, 'models were evaluated on different segments!'

        targets = list(runs[0]['overall'].keys())
        songs = list(runs[0]['per_song'].keys())
        results[label] = {
            'ckpt': str(ckpt),
            'n_segments': runs[0]['n_segments'],
            'n_songs': len(songs),
            'seeds': args.seeds,
            'overall': {t: mean_std([r['overall'][t] for r in runs]) for t in targets},
            'first_k': {t: mean_std([r['first_k'][t] for r in runs]) for t in targets},
            'per_song': {s: {t: mean_std([r['per_song'][s][t] for r in runs]) for t in targets}
                         for s in songs},
        }
        del task
        if device == 'cuda':
            torch.cuda.empty_cache()

    # ---------- report ----------
    targets = list(next(iter(results.values()))['overall'].keys())
    songs = list(next(iter(results.values()))['per_song'].keys())
    first = next(iter(results.values()))
    print(f"\n| {first['n_segments']} segments, {first['n_songs']} songs, "
          f"seeds={args.seeds}; variances conditioned on GT pitch\n")

    def row(title, key):
        print(f'{title:<14}' + ''.join(f'{lab:>22}' for lab in results))
        for t in targets:
            cells = ''.join(f"{results[lab][key][t][0]:>14.4f} ±{results[lab][key][t][1]:.4f}" for lab in results)
            print(f'  {t:<12}{cells}')
        print()

    row('ALL segments', 'overall')
    row(f'first {args.first_k}', 'first_k')

    if len(results) == 2:
        a, b = list(results)
        print(f'per-song R^2 ({b} - {a}), positive = {b} better')
        print(f"{'song':<8}" + ''.join(f'{t:>13}' for t in targets))
        wins = {t: 0 for t in targets}
        for s in songs:
            diffs = []
            for t in targets:
                d = results[b]['per_song'][s][t][0] - results[a]['per_song'][s][t][0]
                wins[t] += d > 0
                diffs.append(d)
            print(f'{s:<8}' + ''.join(f'{d:>+13.4f}' for d in diffs))
        print(f"{'wins':<8}" + ''.join(f'{f"{wins[t]}/{len(songs)}":>13}' for t in targets))

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'\n| results written to {args.out}')


if __name__ == '__main__':
    main()
