"""Conditional recognition retries for the standalone TESTING runner.

The evaluator receives RGB image views and returns the existing inference
outcome dictionaries. Labels and expected answers never enter this module.
"""
from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)
FALLBACK_ORDER = ('contrast', 'both', 'rotations')
RETRY_REASONS = frozenset(('zero detections', 'low confidence'))
VERSION = 'conditional-fallback-v1'


def image_key(image):
    image = image.convert('RGB')
    return hashlib.sha256(f'{image.width}x{image.height}|RGB|'.encode() + image.tobytes()).hexdigest()


def enhance_contrast(image):
    """Normalize the light paper level, then apply the tested mild CLAHE."""
    import cv2

    gray = np.asarray(image.convert('L'))
    background = float(np.percentile(gray, 90))
    normalized = np.clip(gray.astype(np.float32) * 255 / max(background, 1), 0, 255).round().astype(np.uint8)
    if gray.max() == gray.min():
        return Image.fromarray(normalized).convert('RGB')
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(normalized)
    return Image.fromarray(enhanced).convert('RGB')


def rotate_view(image, angle):
    if angle == 0:
        return image
    if angle in (90, 180, 270):
        return image.transpose({90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180,
                                270: Image.Transpose.ROTATE_270}[angle])
    return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor='white')


def _score(outcome):
    prediction = outcome.get('prediction')
    if prediction is not None:
        return float(prediction['confidence'])
    candidates = outcome.get('candidates', outcome.get('detector', {}).get('candidates', []))
    return max((float(c['confidence']) for c in candidates), default=-1.0)


def _select(views):
    # A rejected view can never displace an accepted single-letter view.
    accepted = [v for v in views if v[1].get('accepted')]
    if not accepted:
        return None
    return max(accepted, key=lambda v: (_score(v[1]), -v[0]))


def recognize_batch(images, evaluate, enabled=True):
    """BASE first, then contrast, both, rotations only for eligible misses.

    Accepted BASE results and BASE multiple-letter rejections are final.
    Every fallback stage evaluates all its orientations before choosing the
    strongest accepted view. An exhausted cascade returns the BASE rejection.
    Zero-degree views are reused, so each input needs at most 16 model views.
    """
    images = list(images)
    if not images:
        return []
    base = list(evaluate(images))
    if len(base) != len(images):
        raise ValueError('The evaluator returned a different number of outcomes.')
    selected = list(base)
    method = ['base'] * len(images)
    angle = [0] * len(images)
    traces = [[{'method': 'base', 'angle': 0, 'outcome': result}] for result in base]
    pending = [i for i, result in enumerate(base) if enabled and not result.get('accepted')
               and result.get('reason') in RETRY_REASONS]
    contrasted = {}
    contrast_results = {}
    for stage in FALLBACK_ORDER:
        if not pending:
            break
        jobs = []
        stage_views = {i: [] for i in pending}
        for i in pending:
            if stage in ('contrast', 'both'):
                if i not in contrasted:
                    contrasted[i] = enhance_contrast(images[i])
                original = contrasted[i]
            else:
                original = images[i]
            for rotation in (ANGLES if stage in ('both', 'rotations') else (0,)):
                previous = contrast_results.get(i) if stage == 'both' and rotation == 0 else None
                if stage == 'rotations' and rotation == 0:
                    previous = base[i]
                if previous is not None:
                    stage_views[i].append((rotation, previous))
                else:
                    jobs.append((i, rotation, rotate_view(original, rotation)))
        outcomes = list(evaluate([job[2] for job in jobs])) if jobs else []
        if len(outcomes) != len(jobs):
            raise ValueError('The evaluator returned a different number of outcomes.')
        for (i, rotation, _), result in zip(jobs, outcomes):
            stage_views[i].append((rotation, result))
            if stage == 'contrast':
                contrast_results[i] = result
        remaining = []
        for i in pending:
            views = sorted(stage_views[i], key=lambda v: v[0])
            traces[i].extend({'method': stage, 'angle': a, 'outcome': result} for a, result in views)
            chosen = _select(views)
            if chosen is None:
                remaining.append(i)
            else:
                angle[i], selected[i] = chosen
                method[i] = stage
        pending = remaining
    return [dict(outcome=result, base=base[i], preprocessing={
                'version': VERSION, 'method': method[i], 'angle_degrees': angle[i],
                'recovered': method[i] != 'base' and bool(result.get('accepted')),
                'needs_review': method[i] != 'base' and bool(result.get('accepted')),
                'attempted_methods': list(dict.fromkeys(v['method'] for v in traces[i])),
                'unique_views_requested': 1 + sum(1 for v in traces[i][1:]
                    if not (v['angle'] == 0 and v['method'] in ('both', 'rotations')))},
                trace=traces[i]) for i, result in enumerate(selected)]


class YoloEvaluator:
    """FP32 batched detector evaluator with optional signed-run caching."""
    def __init__(self, model, policy, device='cpu', batch=32, cache=None, on_cache_add=None):
        if batch < 1:
            raise ValueError('batch must be at least 1')
        self.model, self.policy, self.device, self.batch = model, policy, device, batch
        self.model_views = 0
        self.cache, self.on_cache_add = cache, on_cache_add
        self.cache_hits = 0

    def __call__(self, images):
        keys = [image_key(image) for image in images] if self.cache is not None else None
        pending = dict(zip(keys, images)) if keys is not None else None
        if pending is not None:
            pending = {key: image for key, image in pending.items() if key not in self.cache}
            self.cache_hits += len(images) - len(pending)
        work = list(pending.values()) if pending is not None else images
        work_keys = list(pending) if pending is not None else None
        outcomes = []
        for start in range(0, len(work), self.batch):
            chunk = work[start:start + self.batch]
            results = self.model.predict(chunk, imgsz=640, rect=False, conf=self.policy['candidate_floor'],
                iou=self.policy['nms_iou'], agnostic_nms=True, max_det=32, device=self.device,
                quantize=None, augment=False, verbose=False, save=False)
            if getattr(self.model.predictor.model, 'fp16', False):
                raise RuntimeError('Conditional fallback requires FP32 inference.')
            if len(results) != len(chunk):
                raise ValueError('The detector returned a different number of results.')
            batch_outcomes = [outcome_from_result(result, self.policy) for result in results]
            outcomes.extend(batch_outcomes)
            if work_keys is not None:
                for key, result in zip(work_keys[start:start + self.batch], batch_outcomes):
                    self.cache[key] = result
                    if self.on_cache_add is not None:
                        self.on_cache_add(key, result)
            self.model_views += len(chunk)
        return [self.cache[key] for key in keys] if keys is not None else outcomes


def outcome_from_result(result, policy):
    candidates = [dict(class_id=int(c), confidence=float(s), box=[float(x) for x in b])
        for b, s, c in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                           result.boxes.cls.cpu().tolist()) if float(s) >= policy['candidate_floor']]
    if not candidates:
        return dict(accepted=False, reason='zero detections', candidates=[])
    if len(candidates) != 1:
        return dict(accepted=False, reason='multiple letters', candidates=candidates)
    if candidates[0]['confidence'] < policy['acceptance_threshold']:
        return dict(accepted=False, reason='low confidence', candidates=candidates)
    return dict(accepted=True, reason='accepted', prediction=candidates[0], candidates=candidates)
