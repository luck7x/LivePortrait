"""Bounded model-body causal probes; not a trainer, optimizer or postprocessor.

Heavy imports occur only after Linux authorization and workspace preflight.
"""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FRAMES = (0, *range(260, 268))
LIPS = (6, 12, 14, 17, 19, 20)
CASES = ('baseline', 'freeze_lip', 'freeze_pose', 'blur_driver')
ROI = (190, 310, 360, 430)  # x0,y0,x1,y1 in raw512, proxy only
LIMIT = 512 * 2**20


def inside(path, root, exists=False):
    path = Path(path).resolve(strict=exists)
    if not path.is_relative_to(root):
        raise ValueError(f'Path outside workspace: {path}')
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def equal(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return bool(np.array_equal(a, b))


def intervene(base, case):
    """Copy only this run's template; verify every non-intervened field."""
    if case not in ('freeze_lip', 'freeze_pose') or base['n_frames'] != len(FRAMES):
        raise ValueError('Expected a nine-frame fresh template and a freeze case')
    result = copy.deepcopy(base)
    reference = base['motion'][FRAMES.index(264)]
    for item in result['motion'][1:]:
        if case == 'freeze_lip':
            item['exp'][:, LIPS, :] = reference['exp'][:, LIPS, :]
        else:
            for key in ('R', 't', 'scale'):
                item[key] = reference[key].copy()
    restored = copy.deepcopy(result)
    for original, changed in zip(base['motion'][1:], restored['motion'][1:]):
        if case == 'freeze_lip':
            changed['exp'][:, LIPS, :] = original['exp'][:, LIPS, :]
        else:
            for key in ('R', 't', 'scale'):
                changed[key] = original[key].copy()
    if not equal(base, restored) or not equal(base['motion'][0], result['motion'][0]):
        raise RuntimeError('Intervention changed a protected field')
    return result, {'anchor_unchanged': True, 'all_non_target_fields_unchanged': True,
                    'reference_frame': 264, 'fields': ['exp:6,12,14,17,19,20']
                    if case == 'freeze_lip' else ['R', 't', 'scale']}


def proxy(a, b):
    if a.shape != (512, 512, 3) or b.shape != a.shape:
        raise ValueError('Expected raw512 RGB')
    difference = np.abs(a.astype(np.float32) - b.astype(np.float32))
    x0, y0, x1, y1 = ROI
    return {'rgb_mae': float(difference.mean()),
            'mouth_rgb_mae': float(difference[y0:y1, x0:x1].mean())}


class Guard:
    def __init__(self, root, output):
        self.root, self.output = root, output
        self.deadline = time.monotonic() + 300

    def command(self, argv, **kwargs):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('300 second probe limit')
        return subprocess.run(argv, check=True, timeout=remaining, **kwargs)

    def allocated(self, path):
        # GNU du counts allocated blocks, deduplicating hardlinks in one traversal.
        return int(self.command(['du', '-s', '-B1', str(path)],
                                capture_output=True, text=True).stdout.split()[0])

    def check(self, reserve=0):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('300 second probe limit')
        used = self.allocated(self.output) if self.output.exists() else 0
        if used + reserve > LIMIT:
            raise RuntimeError('512MiB output budget exceeded')
        headroom = LIMIT - used
        if self.allocated(self.root) + headroom > 20 * 2**30:
            raise RuntimeError('20GiB allocated workspace budget/headroom exceeded')
        if shutil.disk_usage(self.root).free < headroom:
            raise RuntimeError('Insufficient disk headroom')


def git_state(guard):
    def git(*args):
        return guard.command(['git', '-C', str(REPO), *args], capture_output=True,
                             text=True).stdout.strip()
    pin = os.environ.get('PROBE_CODE_SHA', '')
    head = git('rev-parse', 'HEAD')
    if len(pin) != 40 or pin != head or git('status', '--porcelain', '--untracked-files=all'):
        raise RuntimeError('Require clean Git and exact PROBE_CODE_SHA deployment pin')
    return head


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def run(args):
    if sys.platform != 'linux' or not args.authorize_probe:
        raise RuntimeError('Explicit --authorize-probe on authorized Linux required')
    root = args.workspace.resolve(strict=True)
    inside(REPO, root, True)
    source = inside(args.source, root, True)
    driving = inside(args.driving, root, True)
    output = inside(args.output, root)
    if not source.is_file() or source.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp'):
        raise ValueError('Source must be a photo')
    if not driving.is_file() or driving.suffix.lower() not in ('.mp4', '.mov', '.avi'):
        raise ValueError('Driving must be an original video, never external pickle')
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError('Output must be new and outside clean code worktree')
    for key in ('HOME', 'TMPDIR', 'HF_HOME', 'TORCH_HOME', 'XDG_CACHE_HOME'):
        if not os.environ.get(key):
            raise RuntimeError('Missing isolated environment: ' + key)
        inside(os.environ[key], root, True)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(visible) != 1 or not visible[0].strip() or visible[0].strip() == '-1':
        raise RuntimeError('Expose exactly one previously approved GPU')
    guard = Guard(root, output)
    code_before = git_state(guard)
    guard.check()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    output.mkdir(parents=True)
    report = {'status': 'running', 'kind': 'model_body_causal_probe_not_optimization',
              'frames': FRAMES, 'deliverable_frames': list(FRAMES[1:]),
              'roi_xyxy': ROI, 'gpu': visible[0], 'code_before': code_before,
              'worker_pid': os.getpid(), 'worker_process_group': os.getpgrp(),
              'cases': {}, 'warnings': ['MAE is a proxy, not a tooth quality score',
              'Frozen/reduced motion is not a stability repair',
              'Internal nine-frame inputs include a time jump; never deliver them']}
    report_path = output / 'probe.json'
    write_json(report_path, report)
    before = {}
    old_alarm = signal.getsignal(signal.SIGALRM)

    def timeout(signum, frame):
        raise TimeoutError('300 second probe limit')

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(.001, guard.deadline - time.monotonic()))
    try:
        before = {str(p): sha256(p) for p in (source, driving)}
        report['data_before'] = before
        # The config loads this tracked resource pickle; never follow it outside project.
        for relative in ('src/utils/resources/lip_array.pkl',
                         'src/utils/resources/mask_template.png', 'src/config/models.yaml'):
            inside(REPO / relative, root, True)
        # No Torch/CV2/model import before all preflight checks above.
        sys.path.insert(0, str(REPO))
        import cv2
        import torch
        from PIL import Image
        from src.config.argument_config import ArgumentConfig
        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        import src.live_portrait_pipeline as pipeline_module

        random.seed(20260910)
        np.random.seed(20260910)
        torch.manual_seed(20260910)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one visible CUDA device required')
        torch.cuda.manual_seed_all(20260910)
        torch.backends.cudnn.benchmark = False  # pipeline import sets True; override now
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        cfg = InferenceConfig(flag_relative_motion=True, flag_stitching=True,
            flag_normalize_lip=True, flag_crop_driving_video=False,
            flag_source_video_eye_retargeting=False, flag_eye_retargeting=False,
            flag_lip_retargeting=False, flag_pasteback=False, driving_option='expression-friendly',
            animation_region='all', flag_do_torch_compile=False, device_id=0)
        crop_cfg = CropConfig(device_id=0, det_thresh=0.15)
        weights = [inside(getattr(cfg, 'checkpoint_' + k), root, True) for k in 'FMWGS']
        for p in (cfg.models_config, crop_cfg.insightface_root, crop_cfg.landmark_ckpt_path):
            inside(p, root, True)
        buffalo = inside(Path(crop_cfg.insightface_root) / 'models' / 'buffalo_l', root, True)
        if not list(buffalo.glob('*.onnx')):
            raise RuntimeError('Require pre-existing buffalo_l ONNX files; no downloading')
        for p in Path(crop_cfg.insightface_root).rglob('*'):
            inside(p, root, True)
        report['weights_before'] = {str(p): sha256(p) for p in weights}
        report['config'] = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
                            if f.name not in ('mask_crop', 'lip_array')}
        report['crop_config'] = dataclasses.asdict(crop_cfg)
        report['seed'] = 20260910
        report['cublas_workspace_config'] = os.environ['CUBLAS_WORKSPACE_CONFIG']
        internal = output / 'internal'
        internal.mkdir()
        (internal / 'NOT_FOR_DELIVERY.txt').write_text(
            'Nine-frame rawsubset has f0 -> f260 time jump. Internal only. No full video optimization.\n')

        def decode_selected(path):
            cap = cv2.VideoCapture(str(path))
            try:
                if abs(cap.get(cv2.CAP_PROP_FPS) - 25) > .01:
                    raise ValueError('Expected 25fps original driver')
                frames = []
                for number in FRAMES:
                    if not cap.set(cv2.CAP_PROP_POS_FRAMES, number):
                        raise RuntimeError('Decoder seek failed')
                    ok, bgr = cap.read()
                    if not ok or bgr.shape != (1024, 1024, 3):
                        raise ValueError('Expected original 1024 RGB driver frames')
                    if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - (number + 1)) > .1:
                        raise RuntimeError('Decoder frame-position mismatch')
                    frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                return frames
            finally:
                cap.release()

        frames = decode_selected(driving)
        report['decoded_rgb_hashes'] = dict(zip(map(str, FRAMES), map(array_hash, frames)))
        raw = internal / 'rawsubset_ANCHOR_TIME_JUMP.avi'
        guard.check(40 * 2**20)
        guard.command(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', '1024x1024', '-r', '25', '-i', 'pipe:0',
            '-an', '-c:v', 'ffv1', '-threads', '2', '-filter_threads', '2', '-pix_fmt', 'bgr0', str(raw)],
            input=b''.join(f.tobytes() for f in frames), capture_output=True)
        cap = cv2.VideoCapture(str(raw))
        try:
            for expected in frames:
                ok, bgr = cap.read()
                if not ok or not np.array_equal(expected, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)):
                    raise RuntimeError('FFV1 re-decode changed RGB')
            if cap.read()[0]:
                raise RuntimeError('Unexpected extra rawsubset frame')
        finally:
            cap.release()
        report['rawsubset_rgb_exact'] = True
        del frames
        pipeline = pipeline_module.LivePortraitPipeline(cfg, crop_cfg)
        wrapper = pipeline.live_portrait_wrapper
        original_template = pipeline.make_motion_template
        original_feature = wrapper.extract_feature_3d
        original_warp = wrapper.warp_decode
        original_video = pipeline_module.images2video
        # Suppress only file encoding; execute still constructs final K and calls W/G.
        pipeline_module.images2video = lambda *a, **kw: None
        state = {'base': None, 'feature': None, 'source_hash': None, 'fixed': None}
        outputs = {}

        def cpu(tensor):
            return tensor.detach().cpu().numpy()

        def feature_hook(tensor):
            digest = array_hash(cpu(tensor))
            if state['feature'] is None:
                state['source_hash'] = digest
                state['feature'] = original_feature(tensor)
                guard.check(80 * 2**20)
                np.savez_compressed(internal / 'source_feature.npz',
                                    source_input=cpu(tensor), f_s=cpu(state['feature']))
            elif digest != state['source_hash']:
                raise RuntimeError('prepare_source tensor changed; refusing F reuse')
            return state['feature']

        wrapper.extract_feature_3d = feature_hook
        try:
            for case in CASES:
                guard.check()
                folder = output / case
                folder.mkdir()
                inputs = internal / case / 'inputs'
                inputs.mkdir(parents=True)
                case_driver = inputs / 'rawsubset_ANCHOR_TIME_JUMP.avi'
                os.link(raw, case_driver)
                case_report = {'status': 'running', 'frames': [], 'adjacent_proxy': []}
                report['cases'][case] = case_report
                write_json(report_path, report)
                rendered = []

                def template_hook(tensor, eyes, lips, **kwargs):
                    if tuple(tensor.shape[-2:]) != (256, 256) or tensor.shape[0] != 9:
                        raise RuntimeError('Unexpected actual driving M input')
                    case_report['M_input_before_hash'] = array_hash(cpu(tensor))
                    if case == 'baseline':
                        state['driver_input_hash'] = array_hash(cpu(tensor))
                        template = original_template(tensor, eyes, lips, **kwargs)
                        state['base'] = copy.deepcopy(template)
                    else:
                        if array_hash(cpu(tensor)) != state['driver_input_hash']:
                            raise RuntimeError('Unperturbed driving input differs across cases')
                        if case in ('freeze_lip', 'freeze_pose'):
                            template, invariants = intervene(state['base'], case)
                            case_report['invariants'] = invariants
                        else:
                            # Blur only actual normalized M input. Neither source nor f0 changes.
                            perturbed = tensor.clone()
                            for i in range(1, 9):
                                image = cpu(tensor[i, 0]).transpose(1, 2, 0)
                                small = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
                                small = cv2.GaussianBlur(small, (5, 5), 0)
                                blurred = cv2.resize(small, (256, 256), interpolation=cv2.INTER_LINEAR)
                                perturbed[i, 0] = torch.from_numpy(blurred.transpose(2, 0, 1)).to(tensor)
                            if not torch.equal(tensor[0], perturbed[0]):
                                raise RuntimeError('Blur changed f0 M input')
                            case_report['M_input_after_hash'] = array_hash(cpu(perturbed))
                            template = original_template(perturbed, eyes, lips, **kwargs)
                            if not equal(template['motion'][0], state['base']['motion'][0]):
                                raise RuntimeError('Blur changed anchor motion')
                            case_report['invariants'] = {'anchor_unchanged': True,
                                                        'source_input_unchanged': True}
                            tensor = perturbed
                    case_report['motion_field_hashes'] = [
                        {'frame': FRAMES[i], 'before': {k: array_hash(v) for k, v in
                         state['base']['motion'][i].items()},
                         'after': {k: array_hash(v) for k, v in motion.items()}}
                        for i, motion in enumerate(template['motion'])]
                    arrays = {'M_input': cpu(tensor)}
                    for i, motion in enumerate(template['motion']):
                        for key, value in motion.items():
                            arrays[f'f{FRAMES[i]}_{key}'] = value
                    arrays['c_eyes'] = np.asarray(template['c_eyes_lst'])
                    arrays['c_lip'] = np.asarray(template['c_lip_lst'])
                    guard.check(sum(a.nbytes for a in arrays.values()) + 2**20)
                    np.savez_compressed(inputs.parent / 'motion_numeric.npz', **arrays)
                    # Never allow pipeline tensor in-place operations to mutate cached baseline.
                    return copy.deepcopy(template)

                def warp_hook(f_s, x_s, x_d):
                    index = len(rendered)
                    if index >= 9:
                        raise RuntimeError('Unexpected extra warp call')
                    fixed = {'f_s': array_hash(cpu(f_s)), 'x_s': array_hash(cpu(x_s)),
                             'prepare_source': state['source_hash']}
                    if state['fixed'] is None:
                        state['fixed'] = fixed
                    if fixed != state['fixed']:
                        raise RuntimeError('F/source keypoints differ across cases')
                    result = original_warp(f_s, x_s, x_d)
                    pixels = wrapper.parse_output(result['out'])[0]
                    if pixels.dtype != np.uint8 or pixels.shape != (512, 512, 3):
                        raise RuntimeError('Expected native raw512 uint8 RGB')
                    number = FRAMES[index]
                    guard.check(3 * 2**20)
                    target = inputs.parent if number == 0 else folder
                    Image.fromarray(pixels).save(target / f'f{number:06d}.png')
                    np.savez_compressed(inputs.parent / f'f{number:06d}_finalK.npz',
                                        x_s=cpu(x_s), final_k=cpu(x_d))
                    if case == 'baseline' and number == 263:
                        repeats = [wrapper.parse_output(original_warp(f_s, x_s, x_d)['out'])[0]
                                   for _ in range(2)]
                        case_report['repeat_f263'] = [
                            {'uint8_exact': bool(np.array_equal(pixels, r)), **proxy(pixels, r)}
                            for r in repeats]
                        for j, r in enumerate(repeats):
                            Image.fromarray(r).save(inputs.parent / f'f263_repeat{j + 1}.png')
                    case_report['frames'].append({'frame': number, 'pixel_hash': array_hash(pixels),
                                                  'final_k_hash': array_hash(cpu(x_d)), **fixed})
                    rendered.append(pixels.copy())
                    return result

                pipeline.make_motion_template = template_hook
                wrapper.warp_decode = warp_hook
                arguments = ArgumentConfig(source=str(source), driving=str(case_driver),
                                            output_dir=str(inputs.parent / 'suppressed_video'))
                for key in ('flag_relative_motion', 'flag_stitching', 'flag_normalize_lip',
                            'flag_crop_driving_video', 'flag_source_video_eye_retargeting',
                            'flag_eye_retargeting', 'flag_lip_retargeting', 'flag_pasteback',
                            'driving_option', 'animation_region'):
                    setattr(arguments, key, getattr(cfg, key))
                case_report['arguments'] = dataclasses.asdict(arguments)
                with torch.no_grad():
                    pipeline.execute(arguments)
                if len(rendered) != 9:
                    raise RuntimeError('Probe did not render exactly nine internal frames')
                if case != 'baseline' and not np.array_equal(rendered[0], outputs['baseline'][0]):
                    raise RuntimeError('Anchor rendered pixels differ')
                for i in range(2, 9):
                    case_report['adjacent_proxy'].append({'from': FRAMES[i - 1], 'to': FRAMES[i],
                                                        **proxy(rendered[i - 1], rendered[i])})
                outputs[case] = rendered
                guard.check(16 * 2**20)
                diagnostic = folder / 'diagnostic_f260-f267_8frames_silent.mp4'
                guard.command(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-framerate', '25',
                    '-start_number', '260', '-i', str(folder / 'f%06d.png'), '-frames:v', '8',
                    '-an', '-c:v', 'libx264', '-threads', '2', '-filter_threads', '2', '-crf', '15', '-pix_fmt', 'yuv420p', str(diagnostic)],
                    capture_output=True)
                cap = cv2.VideoCapture(str(diagnostic))
                count = 0
                try:
                    while cap.read()[0]:
                        count += 1
                finally:
                    cap.release()
                if count != 8:
                    raise RuntimeError('Diagnostic video decode count mismatch')
                case_report.update(status='completed', video_decoded_frames=count)
                write_json(report_path, report)
            for case in CASES[1:]:
                report['cases'][case]['vs_baseline_proxy'] = [
                    {'frame': FRAMES[i], **proxy(outputs['baseline'][i], outputs[case][i])}
                    for i in range(1, 9)]
        finally:
            pipeline_module.images2video = original_video
            pipeline.make_motion_template = original_template
            wrapper.extract_feature_3d = original_feature
            wrapper.warp_decode = original_warp
        report['weights_after'] = {str(p): sha256(p) for p in weights}
        if report['weights_before'] != report['weights_after']:
            raise RuntimeError('Weights changed during probe')
        report['data_after'] = {p: sha256(p) for p in before}
        report['code_after'] = git_state(guard)
        if before != report['data_after'] or code_before != report['code_after']:
            raise RuntimeError('Source/driving/code changed during probe')
        guard.check()
        report['allocated_output_bytes'] = guard.allocated(output)
        report['allocated_workspace_bytes'] = guard.allocated(root)
        report['elapsed_seconds'] = 300 - (guard.deadline - time.monotonic())
        report['postflight_verified'] = True
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}',
                      postflight_verified=False)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        write_json(report_path, report)
    return report


def worker(args):
    # Own process group includes ffmpeg/du children; never touch pre-existing processes.
    os.setsid()
    run(args)


def supervised(args):
    """Parent enforces wall time even if a native CUDA call blocks Python signals."""
    if sys.platform != 'linux' or not args.authorize_probe:
        raise RuntimeError('Explicit --authorize-probe on authorized Linux required')
    root = args.workspace.resolve(strict=True)
    output = inside(args.output, root)
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError('Output must be new and outside clean code worktree')
    import multiprocessing
    process = multiprocessing.get_context('fork').Process(target=worker, args=(args,))
    process.start()
    try:
        process.join(299)
        if process.is_alive():
            raise TimeoutError('Hard 300 second probe wall-time limit')
        if process.exitcode:
            raise RuntimeError(f'Probe worker failed with exit code {process.exitcode}; inspect probe.json')
    finally:
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.join(.5)
                if process.is_alive():
                    os.killpg(process.pid, signal.SIGKILL)
                    process.join(.5)
            except ProcessLookupError:
                # Possible immediate cancellation before child setsid().
                process.terminate()
                process.join(.5)
                if process.is_alive():
                    process.kill()
                    process.join(.5)
            # Keep partial artifacts. Do not claim postflight validation after a kill.
            root = args.workspace.resolve(strict=True)
            path = inside(args.output, root) / 'probe.json'
            if path.is_file():
                data = json.loads(path.read_text(encoding='utf-8'))
                data.update(status='failed', error='Supervisor interrupted or wall-time exceeded',
                            postflight_verified=False)
                write_json(path, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'source', 'driving', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--authorize-probe', action='store_true')
    supervised(parser.parse_args())


if __name__ == '__main__':
    main()
