"""Preregistered source-mouth compensation. No training, student or RGB patch."""
import argparse
from contextlib import ExitStack
import copy
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import probe_source_mouth as base
from scripts.snapshot_upper_teeth import array_hash, code_check, inside, require, sha256
from scripts.probe_upper_tail_capacity import bounded_npz

MODES = ('fixed-half', 'fixed-three-quarter', 'dynamic')
DRIVERS = {'clip118': 'd06610401c8de2f19a008142bea4c3123b8617e3b2f38f5ee9bf510ee06c7eda',
           'clip80': '506fda347ecd7a20a79a33df9ebc2493816455c7bdd63c8c2ec0cb28d45928fe'}


def coefficients(ratios, mode):
    r = np.asarray(ratios)
    require(r.shape == (581,) and r.dtype.kind == 'f' and np.isfinite(r).all() and (r >= 0).all(), 'invalid true driving lip ratios')
    require(mode in MODES, 'unregistered mode')
    if mode != 'dynamic':
        return np.full(581, .5 if mode == 'fixed-half' else .75, np.float32)
    z = np.clip((r.astype(np.float64) - .03) / (.25 - .03), 0, 1)
    return (1 - .5 * (z * z * (3 - 2 * z))).astype(np.float32)


class DeltaCursor:
    """Own two fresh storages; never mutate the MLP result, source or model state."""
    def __init__(self, raw, alphas, forbidden):
        self.original = raw.detach().clone()
        self.live = self.original.clone()
        pointers = [v.untyped_storage().data_ptr() for v in (raw, *forbidden)]
        own = [v.untyped_storage().data_ptr() for v in (self.original, self.live)]
        require(own[0] != own[1] and all(p not in pointers for p in own), 'delta alias detected')
        self.alphas = alphas
        self.next_frame = 0
        self.live.copy_(self.original * float(alphas[0]))

    def consumed(self, frame):
        require(frame == self.next_frame and frame < 581, 'nonsequential alpha consumption')
        self.next_frame += 1
        if self.next_frame < 581:
            self.live.copy_(self.original * float(self.alphas[self.next_frame]))


def validate_route(args, report, budget, output):
    # Existing validator checks the ded9 source and unchanged ON configuration only.
    require(base.check_config(report, 'student') == base.TWO_DRIVER_SOURCE, 'wrong fixed source')
    driving = report['ArgumentConfig']['driving']
    require(report['inputs_before'][driving] == DRIVERS[args.driver], 'driver route/hash mismatch')
    if args.driver == 'clip80':
        require(args.selected_for_clip80, 'parent must select one clip118 candidate before clip80')
        for path in budget.rglob('report.json'):
            if path.parent != output:
                prior = base.read_json(path, 32 * 2**20)
                require(not (prior.get('kind') == 'mouth-compensation' and prior.get('driver') == 'clip80'
                             and prior.get('status') == 'completed'), 'clip80 candidate already completed; no additional trial')


def collect_keys(report, arrays, crop, canvas, template, alphas, workspace, budget, torch):
    from src import live_portrait_pipeline as pipeline
    from src.live_portrait_wrapper import LivePortraitWrapper
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.config.argument_config import ArgumentConfig
    from src.utils.retargeting_utils import calc_lip_close_ratio
    env = report['environment']
    torch.backends.cudnn.benchmark = env['cudnn_benchmark']
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = env['tf32']
    torch.backends.cuda.matmul.allow_tf32 = env['tf32']
    torch.use_deterministic_algorithms(env['deterministic_algorithms'])
    cfg = base.configured(InferenceConfig, report['cfg'])
    require(cfg.flag_stitching and cfg.flag_normalize_lip and cfg.flag_relative_motion, 'ON control required')
    cfg.flag_pasteback = False  # only collect K here; original pasteback is in reused renderer
    w = LivePortraitWrapper(cfg)
    models = {k: getattr(w, k) for k in ('appearance_feature_extractor', 'motion_extractor', 'warping_module', 'spade_generator')}
    models.update({'retarget:' + k: v for k, v in w.stitching_retargeting_module.items()})
    for model in models.values():
        model.eval().requires_grad_(False)
    before = base.states(models)
    require(before == report['base_before'], 'original model state differs')
    pipe = pipeline.LivePortraitPipeline.__new__(pipeline.LivePortraitPipeline)
    pipe.live_portrait_wrapper = w
    pipe.cropper = SimpleNamespace(crop_cfg=base.configured(CropConfig, report['cropcfg']),
                                  crop_source_image=lambda *a, **kw: {k: v.copy() for k, v in crop.items()})
    source = w.prepare_source(crop['img_crop_256x256'])  # preserve actual source tensor strides
    require(array_hash(source.cpu().numpy()) == report['snapshot_arrays']['source_input'], 'source input differs')
    fsource = torch.from_numpy(arrays['F']).cuda()
    source_info = []
    original_info, original_transform, original_lip = w.get_kp_info, w.transform_keypoint, w.retarget_lip
    def info(value, *a, **kw):
        require(torch.equal(value, source), 'source M input changed')
        if not source_info:
            source_info.append(original_info(source, *a, **kw))
        return {k: v.clone() for k, v in source_info[0].items()}
    def transform(value):
        xs = original_transform(value)
        require(array_hash(xs.cpu().numpy()) == report['snapshot_arrays']['x_s'], 'source xs differs')
        return xs
    def prepare(image):
        require(np.array_equal(image, crop['img_crop_256x256']), 'source crop differs')
        return source.clone()
    def denied(*a, **kw):
        raise RuntimeError('unexpected pipeline media write')
    class Complete(Exception):
        pass
    source_ratio = float(calc_lip_close_ratio(crop['lmk_crop'][None])[0, 0])
    active = source_ratio >= cfg.lip_normalize_threshold
    traces, final = [], None
    forbidden = [source, fsource] + [v for m in models.values() for v in (*m.parameters(), *m.buffers())]
    for label, schedule in (('alpha1-control', np.ones(581, np.float32)), ('candidate', alphas)):
        keys, cursors, original_outputs = [], [], []
        args = base.configured(ArgumentConfig, report['ArgumentConfig'])
        def lip(xs, ratio):
            require(not cursors, 'expected one source normalization MLP call per pass')
            raw = original_lip(xs, ratio)
            require(raw.dtype == torch.float32 and tuple(raw.shape) == (1, 21, 3) and torch.isfinite(raw).all().item(), 'invalid original lip delta')
            original_outputs.append((raw, array_hash(raw.cpu().numpy())))
            cursor = DeltaCursor(raw, schedule, [xs, ratio, *source_info[0].values(), *forbidden])
            cursors.append(cursor)
            return cursor.live
        def collect(f, xs, k):
            frame = len(keys)
            require(torch.equal(f, fsource) and array_hash(xs.cpu().numpy()) == report['snapshot_arrays']['x_s'], 'F/xs changed')
            value = k.detach().cpu().numpy().copy()
            if label == 'alpha1-control':
                require(np.array_equal(value, arrays['final_k'][frame]), f'alpha1 K regression failed {frame}')
            keys.append(value)  # consume current K before changing the next frame's tensor
            require(len(cursors) == int(active), 'source ratio/normalization branch mismatch')
            if cursors:
                cursor = cursors[0]
                require(torch.equal(cursor.live, cursor.original * float(schedule[frame])), 'wrong current alpha')
                cursor.consumed(frame)
            if frame % 100 == 0:
                base.space(workspace, budget)
                print(f'{label}: collected frame {frame}', flush=True)
            if frame == 580:
                raise Complete()
            return {'out': None}
        with ExitStack() as stack, torch.no_grad():
            for name, value in {'is_template': lambda p: p == args.driving, 'load': lambda p: copy.deepcopy(template),
                'load_image_rgb': lambda p: canvas.copy(), 'dump': denied, 'mkdir': denied, 'images2video': denied}.items():
                stack.enter_context(patch.object(pipeline, name, value))
            for name, value in {'prepare_source': prepare, 'get_kp_info': info, 'transform_keypoint': transform,
                'extract_feature_3d': lambda value: fsource.clone(), 'retarget_lip': lip, 'warp_decode': collect,
                'parse_output': lambda value: [np.zeros((1, 1, 3), np.uint8)]}.items():
                stack.enter_context(patch.object(w, name, value))
            try:
                pipe.execute(args)
            except Complete:
                pass
        require(len(keys) == 581, 'incomplete original pipeline K sequence')
        require(all(array_hash(v.cpu().numpy()) == h for v, h in original_outputs), 'original MLP delta mutated')
        traces.append({'pass': label, 'K_sha256': array_hash(np.stack(keys)), 'normalization_calls': len(cursors),
                       'original_delta_sha256': original_outputs[0][1] if original_outputs else None,
                       'new_storages_only': True, 'consumed': cursors[0].next_frame if cursors else 0})
        final = np.stack(keys)
    require(base.states(models) == before, 'original model mutated during collection')
    return final, models, {'alpha1_K_exact_all581': True, 'source_M_calls': len(source_info),
        'source_ratio': source_ratio, 'source_threshold': cfg.lip_normalize_threshold, 'normalization_active': active,
        'consumption_order': 'original stitching(xs, driving_K) + scaled_lip_delta, then original multiplier',
        'traces': traces}


def worker(args, workspace, snapshot, budget, output):
    manifest, result = {}, {}
    started = time.monotonic()
    code = code_check()
    try:
        report, arrays, manifest = base.prep.load_inputs(snapshot, workspace, manifest)
        validate_route(args, report, budget, output)
        base.verify_blobs(report['code_sha'], (*base.real.CORE, 'src/live_portrait_pipeline.py', 'src/live_portrait_wrapper.py',
            'src/modules/motion_extractor.py', 'src/modules/convnextv2.py', 'src/modules/stitching_retargeting_network.py',
            'src/utils/camera.py', 'src/utils/retargeting_utils.py', 'src/utils/helper.py', 'src/utils/resources/lip_array.pkl',
            'src/config/argument_config.py', 'src/config/crop_config.py'), workspace, manifest)
        template = base.load_template(snapshot / 'driving_motion.npz', report['template'])
        ratios = np.asarray(template['c_lip_lst']).reshape(581)
        alphas = coefficients(ratios, args.mode)
        crop = bounded_npz(snapshot / 'source_crop.npz', 16, set(report['crop_arrays']))
        from PIL import Image
        canvas = np.asarray(Image.open(snapshot / 'source_canvas.png').convert('RGB'))
        require(array_hash(canvas) == report['source_canvas']['array_sha256'], 'source canvas changed')
        base.write_json(output / 'report.json', {'status': 'running', 'input_hashes': manifest})
        torch = base.runtime(report)
        keys, models, trace = collect_keys(report, arrays, crop, canvas, template, alphas, workspace, budget, torch)
        before = base.states(models)
        result = base.render(args, workspace, snapshot, budget, output, report, arrays, keys, crop, canvas, models, None, torch)
        require(base.states(models) == before and code_check() == code and all(sha256(p) == h for p, h in manifest.items()), 'model/input/code mutation')
        require(base.prep.validate_snapshot(arrays) == report['snapshot_arrays'], 'cached source/K mutated')
        np.savez(output / 'alpha_and_K.npz', alpha=alphas, driving_lip_ratio=ratios, final_k=keys)
        result.update(status='completed', kind='mouth-compensation', stage=args.stage, driver=args.driver, mode=args.mode,
            code_sha=code, quality_pass=False, semantic_safety='not established; no outside-teeth RGB guarantee',
            intervention='scaled source-normalization MLP delta; original motion operation order retained; not normalize-off',
            alpha=alphas.tolist(), driver_lip_ratios_sha256=array_hash(ratios), alpha_sha256=array_hash(alphas),
            formula='fixed .5/.75 or 1-.5*smoothstep(clamp((r-.03)/(.25-.03),0,1)); no EMA',
            trace=trace, input_hashes=manifest, source_snapshot_arrays=report['snapshot_arrays'],
            base_before=before, base_after=base.states(models), weights_before=report['weights_before'],
            weights_after={p: sha256(p) for p in report['weights_before']}, source_sha256=base.TWO_DRIVER_SOURCE,
            source_redetected=False, source_F_recomputed=False, student_loaded=False, GT_or_label_input=False,
            worker_wall_seconds=time.monotonic()-started, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            files=base.real.inventory(output), supervisor_verified=False)
        base.write_json(output / 'report.json', result)
    except BaseException as exc:
        result.update(status='failed', stage=args.stage, driver=args.driver, mode=args.mode, quality_pass=False,
                      error=repr(exc), input_hashes=manifest, worker_wall_seconds=time.monotonic()-started, supervisor_verified=False)
        base.write_json(output / 'report.json', result)
        raise


class Parser(argparse.ArgumentParser):
    def parse_args(self, *a, **kw):
        args = super().parse_args(*a, **kw)
        args.stage = 'mouth-compensation:' + args.driver + ':' + args.mode
        args.checkpoint = None
        return args


def parser():
    p = Parser(description=__doc__)
    for name in ('workspace', 'snapshot', 'budget-root', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--driver', choices=tuple(DRIVERS), required=True)
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--selected-for-clip80', action='store_true', help='Parent selected this single mode after clip118 review')
    p.add_argument('--authorize-mouth-compensation', dest='authorize_source_mouth', action='store_true')
    p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    return p


def main():
    # Process-scoped supervisor adapter only. Spawn this entry point, including --_worker.
    # ExitStack restores every adapted global on exceptions; motion hooks have their own stack.
    with ExitStack() as stack:
        stack.enter_context(patch.object(base, 'parser', parser))
        stack.enter_context(patch.object(base, 'worker', worker))
        stack.enter_context(patch.object(base, '__file__', __file__))
        base.main()


if __name__ == '__main__':
    main()
