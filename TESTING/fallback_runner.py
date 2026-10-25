"""Full local tests for conditional fallback in the standalone batch runner."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_test as base_runner
from preprocessing import YoloEvaluator, VERSION, recognize_batch

STATUS = {'accepted': 'accepted', 'zero detections': 'no_detection',
          'multiple letters': 'multiple_detections', 'low confidence': 'low_confidence',
          }
EXTRA_FIELDS = ['recognizer', 'selected_method', 'selected_angle_degrees', 'recovered', 'needs_review',
                'attempted_methods', 'unique_views_requested', 'base_status', 'base_predicted_label_id',
                'base_confidence_percent', 'base_correct', 'base_reason', 'final_reason', 'trace_json']


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def candidate_list(outcome):
    return outcome.get('candidates', outcome.get('detector', {}).get('candidates', []))


def status(outcome):
    return STATUS.get(outcome['reason'], 'error')


def compact(outcome):
    candidates = candidate_list(outcome)
    return {'accepted': bool(outcome['accepted']), 'reason': outcome['reason'], 'status': status(outcome),
            'prediction': outcome.get('prediction'), 'candidates': candidates,
            'matches': outcome.get('matches', []), 'error': outcome.get('error', '')}


def make_row(dataset, record, file_hash, result, labels, recognizer, error=''):
    outcome, baseline, prep = result['outcome'], result['base'], result['preprocessing']
    candidates = candidate_list(outcome)
    top = max(candidates, key=lambda c: c['confidence']) if candidates else None
    prediction = outcome.get('prediction') if outcome['accepted'] else None
    predicted = prediction['class_id'] if prediction else None
    base_prediction = baseline.get('prediction') if baseline['accepted'] else None
    expected = record['expected']
    label = labels[expected] if expected is not None else None
    guess = labels[predicted] if predicted is not None else None
    trace = [{'method': v['method'], 'angle': v['angle'], **compact(v['outcome'])} for v in result['trace']]
    return {'dataset': dataset, 'image_relative_path': record['relative'],
        'source_relative_path': record['source_relative'], 'modification': record['modification'],
        'expected_label_id': expected if expected is not None else '',
        'expected_label': label['transliteration'] if label else '', 'expected_character': label['letter'] if label else '',
        'predicted_label_id': predicted if predicted is not None else '',
        'predicted_label': guess['transliteration'] if guess else '', 'predicted_character': guess['letter'] if guess else '',
        'confidence_percent': 100 * prediction['confidence'] if prediction else '',
        'correct': 'yes' if expected is not None and predicted == expected else 'no',
        'status': status(outcome), 'candidate_count': len(candidates),
        'top_candidate_label': labels[top['class_id']]['transliteration'] if top else '',
        'top_confidence_percent': 100 * top['confidence'] if top else '', 'image_file_sha256': file_hash,
        'source_file_sha256': record['source_file_sha256'] or file_hash,
        **{k: record[k] for k in base_runner.PARAMETERS},
        'candidates_json': json.dumps(candidates), 'error': error or outcome.get('error', ''),
        'recognizer': recognizer, 'selected_method': prep['method'], 'selected_angle_degrees': prep['angle_degrees'],
        'recovered': 'yes' if prep['recovered'] else 'no', 'needs_review': 'yes' if prep['needs_review'] else 'no',
        'attempted_methods': ','.join(prep['attempted_methods']), 'unique_views_requested': prep['unique_views_requested'],
        'base_status': status(baseline), 'base_predicted_label_id': base_prediction['class_id'] if base_prediction else '',
        'base_confidence_percent': 100 * base_prediction['confidence'] if base_prediction else '',
        'base_correct': 'yes' if expected is not None and base_prediction and base_prediction['class_id'] == expected else 'no',
        'base_reason': baseline['reason'], 'final_reason': outcome['reason'], 'trace_json': json.dumps(trace)}


def baseline_rows(rows):
    return [dict(row, status=row['base_status'], predicted_label_id=row['base_predicted_label_id'],
                 correct=row['base_correct']) for row in rows]


def summarize(dataset, rows, destination, labels, info):
    report = base_runner.save_report(dataset, rows, destination, labels, info)
    before, _, _ = base_runner.metrics(baseline_rows(rows), labels)
    report['baseline'] = before
    report['fallback_order'] = ['contrast', 'both', 'rotations']
    report['fallback_rules'] = 'Preserve accepted BASE and multiple-letter rejections; retry zero/low confidence; stop at first accepted stage.'
    report['review_note'] = 'Recovered predictions need review. The browser scanner is unchanged; this flag is recorded in the batch CSV.'
    report['methods'] = dict(Counter(row['selected_method'] for row in rows))
    report['attempts'] = dict(Counter(method for row in rows for method in row['attempted_methods'].split(',')))
    report['base_predictions_changed'] = sum(row['base_status'] == 'accepted' and
        (row['status'] != 'accepted' or str(row['predicted_label_id']) != str(row['base_predicted_label_id'])) for row in rows)
    report['base_multiple_changed'] = sum(row['base_status'] == 'multiple_detections' and
        row['status'] != 'multiple_detections' for row in rows)
    assert report['base_predictions_changed'] == report['base_multiple_changed'] == 0
    report['gained_correct'] = sum(row['base_correct'] != 'yes' and row['correct'] == 'yes' for row in rows)
    report['lost_correct'] = sum(row['base_correct'] == 'yes' and row['correct'] != 'yes' for row in rows)
    report['added_wrong_accepts'] = sum(row['recovered'] == 'yes' and row['correct'] != 'yes' for row in rows)
    (destination / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    text = '# ' + dataset['name'] + ' Conditional Fallback Retest\n\n'
    text += ('Recognizer: ' + info['recognizer'] + '. BASE, contrast, both, then eight rotations. '
             'Accepted BASE predictions and BASE multiple-letter rejections are preserved. '
             'Fallback predictions are flagged for review.\n\n')
    text += '| Method | Accuracy | Macro precision | Macro recall | Macro F1 | Accepted-only correct |\n|---|---:|---:|---:|---:|---:|\n'
    for name, score in [('BASE in this run', before), ('Conditional fallback', report['overall'])]:
        text += '| ' + name + ' | ' + ' | '.join(f'{score[k]:.2f}%' for k in
            ('accuracy_percent', 'macro_precision_percent', 'macro_recall_percent', 'macro_f1_percent', 'accepted_only_accuracy_percent')) + ' |\n'
    text += f"\nAdditional correct: {report['gained_correct']:,}. Additional wrong accepted: {report['added_wrong_accepts']:,}. Lost correct: {report['lost_correct']:,}.\n"
    text += '\nAccuracy includes rejected images. These are image-level recognition scores, not bounding-box mAP. No training or threshold tuning was performed.\n'
    (destination / 'Report.md').write_text(text, encoding='utf-8')
    return report


def run(model_path, labels_path, policy_path, folders, output, batch=32, device='auto',
        cache_path=None, include_controls=True):
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path).resolve()
    labels = base_runner.load_labels(labels_path)
    detector_policy = json.loads(Path(policy_path).read_text(encoding='utf-8'))
    model_hash = base_runner.sha256(model_path)
    if model_hash != detector_policy['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum differs from the saved policy.')
    datasets = [base_runner.inventory(folder, labels) for folder in folders]
    manifest = ROOT / 'data/prepared/manifests/val_negative.jsonl'
    if include_controls and manifest.is_file():
        records = []
        for line in manifest.read_text(encoding='utf-8').splitlines():
            entry = json.loads(line)
            records.append(dict(path=ROOT / entry['image'], relative=entry['image'], source_relative=entry['image'],
                expected=None, modification=entry['kind'], expected_hash=entry.get('sha256', ''), source_file_sha256=entry.get('sha256', ''),
                **{k: '' for k in base_runner.PARAMETERS}))
        datasets.append(dict(name='Negative controls', records=records, folder=ROOT, image_root=ROOT, modified=False))
    elif include_controls:
        print('Repository validation controls are unavailable; testing the four selected folders only.', flush=True)
    for dataset in datasets:
        if dataset['name'] != 'Negative controls' and (output == dataset['folder'] or dataset['folder'] in output.parents):
            raise ValueError('Results must be outside the dataset folders.')
    chosen_device = (0 if torch.cuda.is_available() else 'cpu') if device == 'auto' else (int(device) if str(device).isdigit() else device)
    torch.set_num_threads(6)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = YOLO(str(model_path))
    model.model.float()
    if model.task != 'detect' or len(model.names) != 32 or [model.names[i] for i in range(32)] != [x['letter'] for x in labels]:
        raise ValueError('Model classes do not match labels.json.')
    signature = dict(checkpoint_sha256=model_hash, policy=detector_policy, torch=torch.__version__,
        ultralytics=ultralytics.__version__, precision='FP32', batch_size=batch,
        device_name=torch.cuda.get_device_name(chosen_device) if chosen_device != 'cpu' else 'CPU',
        opencv=cv2.__version__, preprocessing_version=VERSION)
    cache_path = Path(cache_path) if cache_path else output / 'detector_cache.jsonl'
    cache_path = cache_path.resolve()
    for dataset in datasets:
        if dataset['name'] != 'Negative controls' and dataset['folder'] in cache_path.parents:
            raise ValueError('The prediction cache must be outside the dataset folders.')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    signature_path = cache_path.with_suffix('.signature.json')
    if signature_path.exists():
        if json.loads(signature_path.read_text()) != signature:
            raise ValueError('Detector cache settings differ; choose a new cache.')
    elif cache_path.exists():
        raise ValueError('Unsigned cache cannot be reused.')
    signature_path.write_text(json.dumps(signature, indent=2), encoding='utf-8')
    cache = {}
    if cache_path.exists():
        with cache_path.open(encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                cache[entry['key']] = entry['outcome']
    info = dict(signature, checkpoint=str(model_path), labels_sha256=base_runner.sha256(labels_path),
                recognizer='yolo', started_at=timestamp(), status='running', cache_start_views=len(cache),
                fallback_order=['contrast', 'both', 'rotations'], recognition='conditional-fallback',
                validation_controls=include_controls and manifest.is_file())
    info['source_sha256'] = {str(p.relative_to(ROOT)): base_runner.sha256(p) for p in
        [Path(__file__).with_name('preprocessing.py'), Path(__file__).with_name('run_test.py'), Path(__file__).resolve()]}
    info['protected_source_sha256'] = {str(p.relative_to(ROOT)): base_runner.sha256(p) for p in
        [ROOT / 'serve_ui.py', ROOT / 'persian_ocr/inference.py', ROOT / 'persian_ocr/crop_cnn.py',
         ROOT / 'ui/index.html', ROOT / 'ui/app.js', ROOT / 'ui/styles.css'] if p.exists()}
    info_path = output / 'run_info.json'
    if info_path.exists():
        previous = json.loads(info_path.read_text(encoding='utf-8'))
        for key in ('checkpoint_sha256', 'labels_sha256', 'policy', 'recognizer', 'source_sha256', 'batch_size'):
            if previous.get(key) != info.get(key):
                raise ValueError('Saved results use different code/settings: ' + key)
        info['started_at'] = previous['started_at']
    info_path.write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f"Starting YOLO full retest on {signature['device_name']}; {len(cache):,} model views cached.", flush=True)
    reports = []
    started = time.perf_counter()
    with cache_path.open('a', encoding='utf-8') as cache_file, ThreadPoolExecutor(max_workers=6) as pool:
        def add_cache(key, outcome):
            cache_file.write(json.dumps(dict(key=key, outcome=outcome)) + '\n')
        evaluator = YoloEvaluator(model, detector_policy, chosen_device, batch, cache, add_cache)
        for dataset in datasets:
            destination = output / dataset['name']
            destination.mkdir(exist_ok=True)
            csv_path = destination / 'predictions.csv'
            rows = list(csv.DictReader(csv_path.open(encoding='utf-8-sig', newline=''))) if csv_path.exists() else []
            planned = {r['relative']: r for r in dataset['records']}
            completed = {r['image_relative_path'] for r in rows}
            if len(completed) != len(rows) or not completed <= set(planned):
                raise ValueError('Saved rows do not match inventory.')
            for row, current_hash in zip(rows, pool.map(base_runner.sha256, [planned[r['image_relative_path']]['path'] for r in rows])):
                if current_hash != row['image_file_sha256'] or str(planned[row['image_relative_path']]['expected'] if planned[row['image_relative_path']]['expected'] is not None else '') != row['expected_label_id']:
                    raise ValueError('A completed input changed.')
            pending = [r for r in dataset['records'] if r['relative'] not in completed]
            print(f"{dataset['name']}: {len(dataset['records']):,} images; {len(rows):,} complete.", flush=True)
            with csv_path.open('a' if rows else 'w', encoding='utf-8' if rows else 'utf-8-sig', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=base_runner.FIELDS + EXTRA_FIELDS)
                if not rows:
                    writer.writeheader()
                for start in range(0, len(pending), batch):
                    chunk = pending[start:start + batch]
                    decoded = list(pool.map(base_runner.decode, chunk))
                    valid = [i for i, (_, _, _, error) in enumerate(decoded) if not error]
                    results = recognize_batch([decoded[i][0] for i in valid], evaluator)
                    by_index = dict(zip(valid, results))
                    for i, record in enumerate(chunk):
                        _, _, file_hash, error = decoded[i]
                        if error:
                            failure = dict(accepted=False, reason='decode error', error=error)
                            result = dict(outcome=failure, base=failure, trace=[], preprocessing=dict(
                                method='base', angle_degrees=0, recovered=False, needs_review=False,
                                attempted_methods=['base'], unique_views_requested=0))
                        else:
                            result = by_index[i]
                        row = make_row(dataset['name'], record, file_hash, result, labels, 'yolo', error)
                        writer.writerow(row)
                        rows.append(row)
                    handle.flush()
                    cache_file.flush()
                    if len(rows) % 256 < batch or len(rows) == len(dataset['records']):
                        progress = dict(recognizer='yolo', dataset=dataset['name'], completed=len(rows),
                            total=len(dataset['records']), new_detector_views=evaluator.model_views,
                            elapsed_seconds=round(time.perf_counter() - started, 1), updated_at=timestamp())
                        (output / 'progress.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
                        print(f"{dataset['name']}: {len(rows):,}/{len(dataset['records']):,}; {evaluator.model_views:,} fresh detector views; {time.perf_counter()-started:.1f}s", flush=True)
            if dataset['name'] == 'Negative controls':
                controls = []
                for kind in sorted({r['modification'] for r in rows}):
                    group = [r for r in rows if r['modification'] == kind]
                    controls.append(dict(kind=kind, images=len(group), base_false_accepts=sum(r['base_status'] == 'accepted' for r in group),
                        fallback_false_accepts=sum(r['status'] == 'accepted' for r in group)))
                base_runner.write_csv(output / 'negative_controls.csv', controls)
            else:
                report = summarize(dataset, rows, destination, labels, info)
                reports.append(report)
                print(f"REPORT READY {dataset['name']}: BASE {report['baseline']['accuracy_percent']:.2f}%; fallback {report['overall']['accuracy_percent']:.2f}%", flush=True)
    comparison = []
    for report in reports:
        for method, score in [('base', report['baseline']), ('fallback', report['overall'])]:
            comparison.append(dict(dataset=report['dataset'], method=method, **{k: v for k, v in score.items() if k != 'status_counts'}))
    base_runner.write_csv(output / 'comparison.csv', comparison)
    base_runner.write_paired_comparisons(output, labels, reports)
    info.update(status='complete' if all(r['run_status'] == 'complete' for r in reports) else 'incomplete',
        finished_at=timestamp(), elapsed_seconds=round(time.perf_counter() - started, 3),
        new_detector_views=evaluator.model_views, cache_total_views=len(cache),
        inputs=sum(len(d['records']) for d in datasets))
    info_path.write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f"COMPLETE YOLO: {info['inputs']:,} inputs, {evaluator.model_views:,} new detector views.", flush=True)
    return reports
