"""Export snapshot-native contact samples for annotation; no candidate or training."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.snapshot_upper_teeth import (OwnedProcessGroup, array_hash, budget,
    code_check, command, inside, require, sha256, validate_snapshot)
from scripts.probe_upper_tail_capacity import bounded_npz

ROI = (200, 330, 320, 380)


def sample_indices():
    return sorted(set(range(222, 232)) | set(range(248, 260)) |
                  {0, 80, 116, 200, 264, 266, 292, 300, 348, 442, 446, 456, 470, 580})


def predecessor(frame):
    require(type(frame) is int and 0 <= frame < 581, 'invalid frame')
    return max(0, frame - 1)


def needed_indices():
    return sorted(set(sample_indices()) | {predecessor(f) for f in sample_indices()})


def validate_sample(arrays):
    require(set(arrays) == {'feature', 'logits'}, 'sample keys mismatch')
    for key, shape in [('feature', (1, 32, 50, 120)), ('logits', (1, 3, 50, 120))]:
        value = arrays[key]
        require(value.shape == shape and value.dtype == np.float32 and
                np.isfinite(value).all(), 'invalid sample ' + key)
    return {key: array_hash(value) for key, value in arrays.items()}


def check_paths(args):
    workspace = Path(args.workspace).resolve(strict=True)
    require(workspace.is_dir(), 'workspace must be directory')
    inside(ROOT, workspace)
    snapshot = inside(args.snapshot, workspace)
    require(snapshot.is_dir(), 'snapshot must be directory')
    base = inside(args.budget_root, workspace, exists=False)
    output = inside(args.output, workspace, exists=False)
    require(output != base and output.is_relative_to(base), 'output must be below budget-root')
    require(not base.is_relative_to(ROOT) and not ROOT.is_relative_to(base), 'budget-root must be separate from code')
    require(not snapshot.is_relative_to(base) and not base.is_relative_to(snapshot), 'snapshot and budget must be separate')
    require(not Path(args.output).is_symlink() and not Path(args.budget_root).is_symlink(), 'symlink output/budget forbidden')
    return workspace, snapshot, base, output


def load_inputs(snapshot, workspace, before):
    report_path = inside(snapshot / 'report.json', snapshot)
    require(report_path.stat().st_size <= 32 * 2**20, 'report size limit')
    before[str(report_path)] = sha256(report_path)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    require(report['status'] == 'completed' and report['supervisor_verified'] is True and
            report['nframes'] == 581 and report['weights_before'] == report['weights_after'] and
            report['inputs_before'] == report['inputs_after'] and
            report['base_before'] == report['base_after'], 'snapshot not verified/immutable')
    require(re.fullmatch('[0-9a-f]{40}', report['code_sha']), 'invalid snapshot SHA')
    require([item['frame'] for item in report['frames']] == list(range(581)), 'incomplete snapshot frame index')
    for name, expected in report['files'].items():
        require(Path(name).name == name, 'snapshot file must be a basename')
        path = inside(snapshot / name, snapshot)
        actual = sha256(path)
        require(actual == expected, 'snapshot file hash mismatch: ' + name)
        before[str(path)] = actual
    require({'source_snapshot.npz', 'source_crop.npz'} <= set(report['files']), 'missing snapshot assets')
    for manifest in ('inputs_before', 'weights_before'):
        for name, expected in report[manifest].items():
            path = inside(name, workspace)
            actual = sha256(path)
            require(actual == expected, 'input hash mismatch: ' + str(path))
            before[str(path)] = actual
    require(len(report['weights_before']) == 5, 'expected original five weights')
    for key in 'FMWGS':
        path = inside(report['cfg']['checkpoint_' + key], workspace)
        require(str(path) in report['weights_before'], 'missing checkpoint manifest ' + key)
    arrays = bounded_npz(snapshot / 'source_snapshot.npz', 12,
                         {'source_input', 'F', 'x_s', 'final_k'})
    require(validate_snapshot(arrays) == report['snapshot_arrays'], 'snapshot array mismatch')
    crop = bounded_npz(snapshot / 'source_crop.npz', 16, set(report['crop_arrays']))
    for key, value in crop.items():
        meta = report['crop_arrays'][key]
        require(value.dtype.kind in 'buif' and np.isfinite(value).all() and
                list(value.shape) == meta['shape'] and str(value.dtype) == meta['dtype'] and
                array_hash(value) == meta['array_sha256'], 'source crop mismatch: ' + key)
    for frame, item in enumerate(report['frames']):
        require(array_hash(arrays['final_k'][frame]) == item['final_k'], 'final K mismatch')
    # Validate every transitive W/G implementation, not just the edited decoder.
    paths = ['src/config/models.yaml', 'src/modules/spade_generator.py',
             'src/modules/util.py', 'src/modules/warping_network.py', 'src/modules/dense_motion.py']
    for rel in paths:
        path = inside(ROOT / rel, workspace)
        blob = command('git', 'rev-parse', report['code_sha'] + ':' + rel)
        require(blob == command('git', 'rev-parse', 'HEAD:' + rel) ==
                command('git', 'hash-object', str(path)), 'snapshot code blob changed: ' + rel)
        before[str(path)] = sha256(path)
    config = inside(report['cfg']['models_config'], workspace)
    require(command('git', 'hash-object', str(config)) ==
            command('git', 'rev-parse', report['code_sha'] + ':src/config/models.yaml'), 'config blob changed')
    before[str(config)] = sha256(config)
    return report, arrays, before


def worker(workspace, snapshot, base, output):
    started = time.monotonic()
    inputs_before = {}
    try:
        code = code_check()
        report, arrays, inputs_before = load_inputs(snapshot, workspace, inputs_before)
        (output / 'report.json').write_text(json.dumps({'status': 'running',
            'supervisor_verified': False, 'inputs_before': inputs_before}, indent=2) + '\n', encoding='utf-8')
        export(workspace, base, output, snapshot, report, arrays, inputs_before, code, started)
    except BaseException as exc:
        after = {}
        for name in inputs_before:
            try:
                after[name] = sha256(name)
            except OSError as error:
                after[name] = {'error': repr(error)}
        failure = {'status': 'failed', 'error': repr(exc), 'supervisor_verified': False,
                   'inputs_before': inputs_before, 'inputs_after': after,
                   'wall_seconds': time.monotonic() - started}
        (output / 'report.json').write_text(json.dumps(failure, indent=2) + '\n', encoding='utf-8')
        raise


def export(workspace, base, output, snapshot, report, arrays, inputs_before, code, started):
    import torch
    import yaml
    from PIL import Image
    from src.modules.spade_generator import SPADEDecoder
    from src.modules.warping_network import WarpingNetwork

    env = report['environment']
    require(env['deterministic_algorithms'] is True and env['tf32'] is False and
            env['cudnn_benchmark'] is False and report['cfg']['flag_use_half_precision'] is True and
            report['cfg']['flag_do_torch_compile'] is False, 'unsupported numerical config')
    require(str(torch.__version__) == env['torch'] and torch.version.cuda == env['cuda'] and
            np.__version__ == env['numpy'], 'snapshot runtime mismatch')
    # These are the explicit settings of snapshot_upper_teeth, not ambient defaults.
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'snapshot CUBLAS config required')
    torch.manual_seed(env['seed'])
    np.random.seed(env['seed'])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.backends.cudnn.benchmark = env['cudnn_benchmark']
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = env['tf32']
    torch.backends.cuda.matmul.allow_tf32 = env['tf32']
    torch.use_deterministic_algorithms(env['deterministic_algorithms'])
    require(torch.cuda.device_count() == 1, 'one visible GPU required')
    config = inside(report['cfg']['models_config'], workspace)
    params = yaml.safe_load(config.read_text())['model_params']
    models = {'warping_module': WarpingNetwork(**params['warping_module_params']).cuda(),
              'spade_generator': SPADEDecoder(**params['spade_generator_params']).cuda()}
    for name, key in [('warping_module', 'W'), ('spade_generator', 'G')]:
        checkpoint = inside(report['cfg']['checkpoint_' + key], workspace)
        models[name].load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True), strict=True)
        models[name].eval().requires_grad_(False)
    w, g = models['warping_module'], models['spade_generator']
    require(isinstance(g.conv_img, torch.nn.Sequential) and len(g.conv_img) == 2 and
            isinstance(g.conv_img[1], torch.nn.PixelShuffle) and g.conv_img[1].upscale_factor == 2,
            'expected original conv/PixelShuffle2 tail')

    def states():
        return {name: {kind: {n: array_hash(t.detach().cpu().numpy()) for n, t in values}
                      for kind, values in [('parameters', model.named_parameters()), ('buffers', model.named_buffers())]}
                for name, model in models.items()}

    before = states()
    require(before == {name: report['base_before'][name] for name in models}, 'loaded W/G state differs from snapshot')
    require(all(not p.requires_grad for model in models.values() for p in model.parameters()), 'base not frozen')
    source_f = torch.from_numpy(arrays['F']).cuda()
    xs = torch.from_numpy(arrays['x_s']).cuda()
    previous_crop = None
    previous_index = None
    records, hashes = [], {}
    samples = set(sample_indices())
    # Only the preceding cropped feature persists. No full H sequence or RGB history.
    with torch.no_grad():
        for frame in needed_indices():
            budget(workspace, base)
            k = torch.from_numpy(arrays['final_k'][frame]).cuda()
            with torch.autocast('cuda', dtype=torch.float16):
                warped = w(source_f, kp_driving=k, kp_source=xs)['out']
                h = g.forward_features(warped)
                require(tuple(h.shape) == (1, 64, 256, 256) and h.dtype == torch.float16, 'unexpected H')
                activated = torch.nn.functional.leaky_relu(h, 2e-1)
                logits = g.conv_img[1](g.conv_img[0](activated))
                image = torch.sigmoid(logits).float()
                feature = torch.nn.functional.pixel_shuffle(activated, 2)[:, :, 330:380, 200:320].float().cpu().numpy().copy()
                log_crop = logits[:, :, 330:380, 200:320].float().cpu().numpy().copy()
            raw = (image.cpu().numpy()[0].transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
            raw_hash = array_hash(raw)
            require(raw_hash == report['frames'][frame]['raw512'], 'raw512 exact mismatch at frame ' + str(frame))
            if frame in samples:
                prev = predecessor(frame)
                require(frame == 0 or previous_index == prev, 'missing actual predecessor')
                values = {'feature': np.concatenate((feature, feature if frame == 0 else previous_crop), axis=1),
                          'logits': log_crop}
                name = f'sample_f{frame:04d}.npz'
                hashes[name] = validate_sample(values)
                np.savez(output / name, **values)
                Image.fromarray(raw).save(output / f'raw512_f{frame:04d}.png')
                records.append({'frame': frame, 'previous': prev, 'raw512': raw_hash,
                                'final_k': array_hash(arrays['final_k'][frame]),
                                'previous_k': array_hash(arrays['final_k'][prev])})
            previous_crop, previous_index = feature, frame
            del h, activated, logits, image, warped, raw, log_crop
    after = states()
    inputs_after = {name: sha256(name) for name in inputs_before}
    require(before == after and inputs_before == inputs_after, 'base/input mutation')
    require(code_check() == code, 'code changed')
    result = {'status': 'completed', 'kind': 'annotation samples only; no optimized video or candidate',
              'snapshot_code_sha': report['code_sha'], 'current_code': code,
              'snapshot': str(snapshot), 'frame_list': sample_indices(), 'computed_frames': needed_indices(),
              'previous_indices': [predecessor(f) for f in sample_indices()], 'frames': records,
              'ROI': list(ROI), 'arrayhashes': hashes, 'sourceF_K_hashes': report['snapshot_arrays'],
              'inputs_before': inputs_before, 'inputs_after': inputs_after,
              'inputweight_hashes': report['weights_before'], 'base_before': before, 'base_after': after,
              'base_frozen': True, 'base_eval': all(not m.training for m in models.values()),
              'environment': env, 'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'],
              'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
              'wall_seconds': time.monotonic() - started, 'supervisor_verified': False,
              'files': {p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name != 'report.json'},
              **budget(workspace, base)}
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'snapshot', 'budget-root', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--authorize-contact', action='store_true')
    p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    return p


def main():
    started = time.monotonic()
    args = parser().parse_args()
    require(sys.platform == 'linux' and args.authorize_contact, 'Linux and explicit authorization required')
    workspace, snapshot, base, output = check_paths(args)
    for key in ('HOME', 'TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'HF_HOME', 'CUDA_CACHE_PATH'):
        require(bool(os.environ.get(key)) and inside(os.environ[key], workspace).is_dir(), 'missing isolated ' + key)
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'disable bytecode writes')
    require(re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', os.environ.get('CUDA_VISIBLE_DEVICES', '')), 'one explicit GPU UUID required')
    code_check()
    if args._worker:
        require(os.environ.get('CONTACT_PREPARE_PARENT') == str(os.getppid()), 'worker must be supervised')
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
        worker(workspace, snapshot, base, output)
        return
    require(not base.exists() and not output.exists(), 'new budget-root/output required; never delete artifacts')
    initial = budget(workspace, base)
    output.mkdir(parents=True)
    owned = timer = process = None
    status, error = 'failed', None
    try:
        env = dict(os.environ, CONTACT_PREPARE_PARENT=str(os.getpid()), CUBLAS_WORKSPACE_CONFIG=':4096:8')
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[key] = '4'
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--_worker'],
                                   cwd=ROOT, env=env, start_new_session=True)
        owned = OwnedProcessGroup(process)
        timer = threading.Timer(max(0, 600 - (time.monotonic() - started)), owned.kill)
        timer.daemon = True
        timer.start()
        while not owned.exited():
            require(time.monotonic() - started < 600, '600s hard timeout')
            budget(workspace, base)
            time.sleep(.5)
        owned.finish()
        require(process.returncode == 0 and time.monotonic() - started < 600, 'worker failed/timed out; partial export rejected')
        report_path = output / 'report.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        report.update(supervisor_verified=True, wall_seconds=time.monotonic() - started,
                      budget_before=initial, budget_after=budget(workspace, base))
        report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        status = 'completed'
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        if timer is not None:
            timer.cancel()
        if owned is not None:
            owned.finish()
        record = {'status': status, 'error': error, 'returncode': None if process is None else process.returncode,
                  'wall_seconds': time.monotonic() - started, 'hard_limit_seconds': 600}
        (output / 'supervisor.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        if status != 'completed':
            report_path = output / 'report.json'
            failure = {}
            if report_path.is_file():
                try:
                    failure = json.loads(report_path.read_text(encoding='utf-8'))
                except (ValueError, OSError):
                    pass
            if 'inputs_before' in failure and 'inputs_after' not in failure:
                failure['inputs_after'] = {}
                for name in failure['inputs_before']:
                    try:
                        failure['inputs_after'][name] = sha256(name)
                    except OSError as exc:
                        failure['inputs_after'][name] = {'error': repr(exc)}
            failure.update(status='failed', supervisor_verified=False, supervisor=record,
                           kind='failed export; no candidate')
            report_path.write_text(json.dumps(failure, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
