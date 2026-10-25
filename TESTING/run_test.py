"""Evaluate labeled Persian image folders with the supplied YOLO checkpoint.

Run without arguments for native folder selection, or pass four folder paths.
Every image receives a CSV row; rejection counts as unsuccessful recognition.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image, ImageOps

NAMES = ['DS1_Persian', 'DS1_Persian_Smoke', 'DS2_Persian_Hand', 'DS2_Persian_Hand_Smoke']
ALIASES = {'aleph':'alef','theh':'seh','che':'cheh','he jimi':'heh-jimi','khe':'kheh',
           'daal':'dal','zaal':'zal','re':'reh','ze':'zeh','zhe':'zheh','taa':'ta',
           'zaa':'za','ayn':'eyn','ghayn':'gheyn','he':'heh','yaa':'yeh'}
PARAMETERS = ['rotation_angle_degrees','blur_radius_pixels','flip_type','brightness_factor',
              'brightness_change_percent','contrast_factor','contrast_change_percent']
FIELDS = ['dataset','image_relative_path','source_relative_path','modification',
          'expected_label_id','expected_label','expected_character','predicted_label_id',
          'predicted_label','predicted_character','confidence_percent','correct','status',
          'candidate_count','top_candidate_label','top_confidence_percent','image_file_sha256',
          'source_file_sha256', *PARAMETERS, 'candidates_json','error']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_labels(path):
    labels=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    assert len(labels)==32 and {row['id'] for row in labels}==set(range(32))
    return sorted(labels,key=lambda row:row['id'])


def canonical(value, by_name):
    value=ALIASES.get(str(value).strip().casefold(),str(value).strip().casefold())
    if value not in by_name:
        raise ValueError(f'Unknown expected label: {value!r}')
    return value


def inventory(folder,labels):
    folder=Path(folder).resolve()
    if not folder.is_dir(): raise FileNotFoundError(folder)
    by_name={row['transliteration']:row for row in labels}
    metadata=folder/'modifications.csv'
    records=[]
    image_root=folder
    if metadata.is_file():
        rows=list(csv.DictReader(metadata.open(encoding='utf-8-sig',newline='')))
        if not rows: raise ValueError(f'Empty modifications.csv in {folder}')
        first=Path(rows[0]['output_relative_path'])
        candidates=[root for root in [folder,*[p for p in folder.iterdir() if p.is_dir()]] if (root/first).is_file()]
        if len(candidates)!=1: raise ValueError(f'Cannot identify image root in {folder}')
        image_root=candidates[0]
        for row in rows:
            relative=Path(row['output_relative_path'])
            path=(image_root/relative).resolve()
            if image_root not in path.parents: raise ValueError('Invalid manifest path.')
            name=canonical(row['letter'],by_name)
            if int(row['label_id'])!=by_name[name]['id']: raise ValueError('Manifest label ID mismatch.')
            records.append({'path':path,'relative':relative.as_posix(),'source_relative':row['source_relative_path'],
                            'expected':by_name[name]['id'],'modification':row['modification'],
                            'expected_hash':row.get('output_file_sha256',''),
                            'source_file_sha256':row.get('source_file_sha256',''),
                            **{key:row.get(key,'') for key in PARAMETERS}})
    else:
        paths=sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg'))
        font_layout=bool(paths) and all('_' in p.stem and p.stem.split('_',1)[1].casefold()==p.parent.name.casefold() for p in paths)
        for path in paths:
            relative=path.relative_to(folder)
            name=canonical(path.stem.split('_',1)[0] if font_layout else relative.parts[0],by_name)
            records.append({'path':path,'relative':relative.as_posix(),'source_relative':relative.as_posix(),
                            'expected':by_name[name]['id'],'modification':'unmodified','expected_hash':'',
                            'source_file_sha256':'',**{key:'' for key in PARAMETERS}})
    actual={p.resolve() for p in image_root.rglob('*') if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg')}
    planned=[row['path'] for row in records]
    if not records or len(set(planned))!=len(planned) or set(planned)!=actual:
        raise ValueError(f'Image inventory and manifest do not match: {folder}')
    counts=Counter(row['expected'] for row in records)
    if set(counts)!=set(range(32)): raise ValueError(f'The test set must include all 32 labels: {folder}')
    return {'name':folder.name,'folder':folder,'image_root':image_root,'records':records,
            'modified':metadata.is_file(),'expected_counts':dict(counts)}


def decode(record):
    try:
        data=record['path'].read_bytes()
        file_hash=hashlib.sha256(data).hexdigest()
        if record['expected_hash'] and record['expected_hash']!=file_hash:
            raise ValueError('Image checksum differs from modifications.csv.')
        if not data or len(data)>10*1024*1024: raise ValueError('Image exceeds the app 10 MB upload limit.')
        with Image.open(io.BytesIO(data)) as original:
            if original.format not in ('PNG','JPEG'): raise ValueError('Only PNG/JPEG images are accepted by the app.')
            if original.width*original.height>16_000_000 or min(original.size)<16:
                raise ValueError('Image dimensions are outside app limits.')
            original.load()
            rgba=ImageOps.exif_transpose(original).convert('RGBA')
        image=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba).convert('RGB')
        key=hashlib.sha256(f'{image.width}x{image.height}|RGB|'.encode()+image.tobytes()).hexdigest()
        return image,key,file_hash,''
    except Exception as error:
        return None,'','',f'{type(error).__name__}: {error}'


def classify_result(result,policy):
    candidates=[{'label_id':int(label),'confidence':float(confidence)}
                for label,confidence in zip(result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist())
                if float(confidence)>=policy['candidate_floor']]
    candidates.sort(key=lambda row:-row['confidence'])
    if not candidates: return {'status':'no_detection','predicted':None,'candidates':candidates}
    if len(candidates)>1: return {'status':'multiple_detections','predicted':None,'candidates':candidates}
    if candidates[0]['confidence']<policy['acceptance_threshold']:
        return {'status':'low_confidence','predicted':None,'candidates':candidates}
    return {'status':'accepted','predicted':candidates[0]['label_id'],'candidates':candidates}


def metrics(rows,labels):
    matrix=np.zeros((32,33),dtype=np.int64)
    statuses=Counter()
    for row in rows:
        predicted=int(row['predicted_label_id']) if row['status']=='accepted' else 32
        matrix[int(row['expected_label_id']),predicted]+=1
        statuses[row['status']]+=1
    tp=np.diag(matrix[:,:32]).astype(float)
    fp=matrix[:,:32].sum(axis=0)-tp
    fn=matrix.sum(axis=1)-tp
    precision=np.divide(tp,tp+fp,out=np.zeros(32),where=(tp+fp)!=0)
    recall=np.divide(tp,tp+fn,out=np.zeros(32),where=(tp+fn)!=0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(32),where=(precision+recall)!=0)
    total=len(rows)
    correct=int(tp.sum())
    accepted=statuses['accepted']
    result={'images':total,'correct':correct,'incorrect_or_rejected':total-correct,
            'accuracy_percent':100*correct/total if total else 0,
            'macro_precision_percent':100*float(precision.mean()),
            'macro_recall_percent':100*float(recall.mean()),'macro_f1_percent':100*float(f1.mean()),
            'accepted_percent':100*accepted/total if total else 0,
            'rejection_percent':100*(total-accepted)/total if total else 0,
            'accepted_only_accuracy_percent':100*correct/accepted if accepted else 0,
            'technical_errors':statuses['error'],'status_counts':dict(statuses)}
    per_letter=[{'label_id':label['id'],'letter':label['transliteration'],'character':label['letter'],
                 'images':int(matrix[i].sum()),'correct':int(tp[i]),
                 'precision_percent':100*float(precision[i]),'recall_percent':100*float(recall[i]),
                 'f1_percent':100*float(f1[i])} for i,label in enumerate(labels)]
    return result,per_letter,matrix


def save_report(dataset,rows,out,labels,run_info):
    out.mkdir(parents=True,exist_ok=True)
    overall,per_letter,matrix=metrics(rows,labels)
    groups=[]
    group_rows=[('processed_all' if dataset['modified'] else 'unmodified',rows)]
    if dataset['modified']:
        for name in ('rotation','flip','rotationFlip','blur','brightnessContrast'):
            subset=[row for row in rows if row['modification']==name]
            if subset: group_rows.append((name,subset))
    for name,subset in group_rows:
        score,_,_=metrics(subset,labels)
        groups.append({'condition':name,**{key:value for key,value in score.items() if key!='status_counts'}})
    write_csv(out/'metrics.csv',groups)
    write_csv(out/'per_letter.csv',per_letter)
    names=[label['transliteration'] for label in labels]
    confusion=[{'expected_label':names[i],**dict(zip(names+['rejected_or_error'],map(int,matrix[i])))} for i in range(32)]
    write_csv(out/'confusion_matrix.csv',confusion)
    report={'dataset':dataset['name'],'folder':str(dataset['folder']),'image_root':str(dataset['image_root']),
            'run_status':'complete' if not overall['technical_errors'] else 'incomplete',
            'modified':dataset['modified'],'overall':overall,'conditions':groups,'per_letter':per_letter,
            'model':run_info,'training_independence':'Not established from the available provenance.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    text=f"# {dataset['name']} Model Test Report\n\n"
    text+=f"Tested {len(rows):,} images using the supplied checkpoint. Correct predictions require one detection with confidence at least 80%. Rejected inputs remain in the denominator.\n\n"
    text+='| Condition | Images | Accuracy | Macro precision | Macro recall | Macro F1 |\n|---|---:|---:|---:|---:|---:|\n'
    for score in groups:
        text+=f"| {score['condition']} | {score['images']:,} | {score['accuracy_percent']:.2f}% | {score['macro_precision_percent']:.2f}% | {score['macro_recall_percent']:.2f}% | {score['macro_f1_percent']:.2f}% |\n"
    text+=f"\nAccepted: {overall['accepted_percent']:.2f}%. Rejected or failed: {overall['rejection_percent']:.2f}%. Technical errors: {overall['technical_errors']}.\n"
    text+='\nScores describe these files under the saved recognition policy. Independence from model training data has not been established.\n'
    (out/'Report.md').write_text(text,encoding='utf-8')
    return report


def run(model_path,labels_path,policy_path,folders,output,batch=32,device='auto'):
    import torch
    import ultralytics
    from ultralytics import YOLO
    model_path,output=Path(model_path).resolve(),Path(output).resolve()
    if not model_path.is_file(): raise FileNotFoundError('The supplied model file was not found.')
    labels=load_labels(labels_path)
    policy=json.loads(Path(policy_path).read_text(encoding='utf-8'))
    model_hash=sha256(model_path)
    if policy.get('checkpoint_sha256') and model_hash!=policy['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum does not match this test configuration.')
    datasets=[inventory(folder,labels) for folder in folders]
    if len({dataset['name'] for dataset in datasets})!=len(datasets): raise ValueError('Dataset folder names must be distinct.')
    for dataset in datasets:
        if output==dataset['folder'] or dataset['folder'] in output.parents:
            raise ValueError('Results must be saved outside the dataset folders.')
    chosen_device=(0 if torch.cuda.is_available() else 'cpu') if device=='auto' else (int(device) if str(device).isdigit() else device)
    torch.set_num_threads(6)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    model=YOLO(str(model_path))
    model.model.float()
    if model.task!='detect' or [model.names.get(i) for i in range(32)]!=[label['letter'] for label in labels] or len(model.names)!=32:
        raise ValueError('Model classes do not exactly match labels.json.')
    info={'checkpoint':str(model_path),'checkpoint_sha256':model_hash,'labels_sha256':sha256(labels_path),
          'policy':policy,'torch':torch.__version__,'ultralytics':ultralytics.__version__,
          'device':str(chosen_device),'device_name':torch.cuda.get_device_name(chosen_device) if chosen_device!='cpu' else 'CPU',
          'precision':'FP32','batch_size':batch,'started_at':datetime.now(timezone(timedelta(hours=8))).isoformat()}
    output.mkdir(parents=True,exist_ok=True)
    info_path=output/'run_info.json'
    signature={key:info[key] for key in ('checkpoint_sha256','labels_sha256','policy','torch','ultralytics','device','device_name','precision','batch_size')}
    if info_path.exists():
        previous=json.loads(info_path.read_text(encoding='utf-8'))
        if any(previous.get(key)!=value for key,value in signature.items()):
            raise ValueError('Existing results were made with different model/settings. Choose a new output directory.')
        info=previous
    info_path.write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    cache={}
    cache_path=output/'prediction_cache.jsonl'
    if cache_path.is_file():
        for line in cache_path.read_text(encoding='utf-8').splitlines():
            try:
                entry=json.loads(line); cache[entry['key']]=entry['outcome']
            except json.JSONDecodeError: pass
    reports=[]
    print(f"Using {info['device_name']}; supplied checkpoint SHA256 {model_hash}",flush=True)
    with cache_path.open('a',encoding='utf-8') as cache_file, ThreadPoolExecutor(max_workers=8) as pool:
        for dataset in datasets:
            started=time.perf_counter()
            destination=output/dataset['name']
            destination.mkdir(exist_ok=True)
            csv_path=destination/'predictions.csv'
            old_rows=[]
            if csv_path.is_file():
                with csv_path.open(encoding='utf-8-sig',newline='') as handle:
                    old_rows=list(csv.DictReader(handle))
            records=dataset['records']
            completed={row['image_relative_path'] for row in old_rows}
            planned={row['relative'] for row in records}
            if not completed<=planned: raise ValueError('Existing predictions do not match current dataset inventory.')
            if len(completed)!=len(old_rows): raise ValueError('Duplicate or partial rows in saved predictions.')
            if old_rows:
                by_relative={record['relative']:record for record in records}
                current_hashes=pool.map(sha256,[by_relative[row['image_relative_path']]['path'] for row in old_rows])
                for row,current_hash in zip(old_rows,current_hashes):
                    expected=by_relative[row['image_relative_path']]['expected']
                    if current_hash!=row['image_file_sha256'] or int(row['expected_label_id'])!=expected:
                        raise ValueError('A completed input changed. Choose a new results folder.')
            pending=[record for record in records if record['relative'] not in completed]
            rows=list(old_rows)
            print(f"Testing {dataset['name']}: {len(records):,} images; {len(completed):,} already recorded.",flush=True)
            with csv_path.open('a' if old_rows else 'w',newline='',encoding='utf-8' if old_rows else 'utf-8-sig') as handle:
                writer=csv.DictWriter(handle,fieldnames=FIELDS)
                if not old_rows: writer.writeheader()
                for offset in range(0,len(pending),batch):
                    chunk=pending[offset:offset+batch]
                    decoded=list(pool.map(decode,chunk))
                    unique={key:image for image,key,_,error in decoded if not error and key not in cache}
                    if unique:
                        keys=list(unique)
                        predictions=model.predict(list(unique.values()),imgsz=640,rect=False,conf=policy['candidate_floor'],
                                                  iou=policy['nms_iou'],agnostic_nms=True,max_det=32,device=chosen_device,
                                                  augment=False,verbose=False,save=False)
                        if getattr(model.predictor.model,'fp16',False):
                            raise RuntimeError('This test requires FP32 inference.')
                        assert len(predictions)==len(keys)
                        for key,prediction in zip(keys,predictions):
                            outcome=classify_result(prediction,policy)
                            cache[key]=outcome
                            cache_file.write(json.dumps({'key':key,'outcome':outcome})+'\n')
                        cache_file.flush()
                    for record,(_,key,file_hash,error) in zip(chunk,decoded):
                        outcome=cache[key] if not error else {'status':'error','predicted':None,'candidates':[]}
                        predicted=outcome['predicted']
                        candidates=outcome['candidates']
                        top=candidates[0] if candidates else None
                        true=labels[record['expected']]
                        guess=labels[predicted] if predicted is not None else None
                        row={'dataset':dataset['name'],'image_relative_path':record['relative'],
                             'source_relative_path':record['source_relative'],'modification':record['modification'],
                             'expected_label_id':true['id'],'expected_label':true['transliteration'],'expected_character':true['letter'],
                             'predicted_label_id':predicted if predicted is not None else '',
                             'predicted_label':guess['transliteration'] if guess else '',
                             'predicted_character':guess['letter'] if guess else '',
                             'confidence_percent':round(100*top['confidence'],6) if guess else '',
                             'correct':'yes' if predicted==true['id'] else 'no','status':outcome['status'],
                             'candidate_count':len(candidates),'top_candidate_label':labels[top['label_id']]['transliteration'] if top else '',
                             'top_confidence_percent':round(100*top['confidence'],6) if top else '',
                             'image_file_sha256':file_hash,'source_file_sha256':record['source_file_sha256'] or file_hash,
                             **{key:record[key] for key in PARAMETERS},'candidates_json':json.dumps(candidates),'error':error}
                        writer.writerow(row); rows.append(row)
                    handle.flush()
                    if (len(rows)//500)>((len(rows)-len(chunk))//500) or len(rows)==len(records):
                        print(f"{dataset['name']}: {len(rows):,}/{len(records):,} images ({time.perf_counter()-started:.1f}s)",flush=True)
            assert len(rows)==len(records)
            report=save_report(dataset,rows,destination,labels,info)
            report['elapsed_seconds']=round(time.perf_counter()-started,3)
            (destination/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
            reports.append(report)
            print(f"REPORT READY {dataset['name']}: accuracy {report['overall']['accuracy_percent']:.2f}%; macro F1 {report['overall']['macro_f1_percent']:.2f}%",flush=True)
    comparison=[{'dataset':report['dataset'],**{k:v for k,v in report['overall'].items() if k!='status_counts'}} for report in reports]
    write_csv(output/'comparison.csv',comparison)
    write_paired_comparisons(output,labels,reports)
    return reports


def write_paired_comparisons(output,labels,reports):
    paired=[]
    for baseline_name,modified_name in (('DS1_Persian','DS1_Persian_Smoke'),('DS2_Persian_Hand','DS2_Persian_Hand_Smoke')):
        if not (output/baseline_name/'predictions.csv').is_file() or not (output/modified_name/'predictions.csv').is_file(): continue
        originals=list(csv.DictReader((output/baseline_name/'predictions.csv').open(encoding='utf-8-sig')))
        modified=list(csv.DictReader((output/modified_name/'predictions.csv').open(encoding='utf-8-sig')))
        by_source={row['source_relative_path']:row for row in originals}
        for name in sorted({row['modification'] for row in modified}):
            subset=[row for row in modified if row['modification']==name]
            matching=[by_source[row['source_relative_path']] for row in subset]
            assert all(raw['image_file_sha256']==changed['source_file_sha256'] and raw['expected_label_id']==changed['expected_label_id']
                       for raw,changed in zip(matching,subset)), 'Paired original identity mismatch.'
            raw,_,_=metrics(matching,labels); changed,_,_=metrics(subset,labels)
            paired.append({'dataset':modified_name,'modification':name,'images':len(subset),
                           **{f'unmodified_{key}':raw[key] for key in ('accuracy_percent','macro_precision_percent','macro_recall_percent','macro_f1_percent')},
                           **{f'modified_{key}':changed[key] for key in ('accuracy_percent','macro_precision_percent','macro_recall_percent','macro_f1_percent')},
                           'accuracy_change_percentage_points':changed['accuracy_percent']-raw['accuracy_percent'],
                           'macro_f1_change_percentage_points':changed['macro_f1_percent']-raw['macro_f1_percent']})
    if paired: write_csv(output/'paired_comparison.csv',paired)


def pick_folders():
    import tkinter as tk
    from tkinter import filedialog
    root=tk.Tk(); root.withdraw()
    folders=[]
    try:
        for name in NAMES:
            folder=filedialog.askdirectory(title=f'Select {name}',initialdir=Path.home()/'Downloads')
            if not folder: raise RuntimeError('Folder selection cancelled.')
            folders.append(folder)
    finally: root.destroy()
    return folders


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders',nargs='*')
    parser.add_argument('--model',type=Path,help='Path to the externally supplied trained YOLO checkpoint.')
    parser.add_argument('--labels',type=Path,default=Path(__file__).with_name('labels.json'))
    parser.add_argument('--policy',type=Path,default=Path(__file__).with_name('decision.json'))
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'Runs'/datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--recognition',choices=('base','fallback'),default='fallback',
                        help='Conditional fallback is the default; base reproduces the earlier detector-only test.')
    parser.add_argument('--cache',type=Path,help='Optional signed detector-view cache shared by local retests.')
    parser.add_argument('--no-controls',action='store_true',help='Skip the repository validation negative controls.')
    parser.add_argument('--no-docx',action='store_true',help='Write CSV/JSON/Markdown without the optional Word report.')
    args=parser.parse_args()
    if args.batch < 1:
        parser.error('--batch must be at least 1.')
    if args.folders and len(args.folders) != 4:
        parser.error('Pass all four dataset folders, or omit them to use the folder pickers.')
    model_path = args.model
    if model_path is None:
        candidate = Path(__file__).resolve().parent.parent / 'yolo11n.pt'
        if candidate.is_file():
            model_path = candidate
        elif args.folders:
            parser.error('Supply --model with the path to your trained yolo11n.pt file.')
        else:
            import tkinter as tk
            from tkinter import filedialog
            window = tk.Tk()
            window.withdraw()
            try:
                selected = filedialog.askopenfilename(title='Select the supplied trained YOLO checkpoint',
                    filetypes=[('PyTorch checkpoint', '*.pt')])
            finally:
                window.destroy()
            if not selected:
                parser.exit(1, 'Model selection cancelled. No images were tested.\n')
            model_path = Path(selected)
    folders = args.folders or pick_folders()
    expected_names = NAMES
    if [Path(folder).name for folder in folders] != expected_names:
        parser.error('Select the four folders in this order: ' + ', '.join(expected_names))
    if args.recognition == 'fallback':
        from fallback_runner import run as run_fallback
        run_fallback(model_path,args.labels,args.policy,folders,args.output,batch=args.batch,device=args.device,
                     cache_path=args.cache,include_controls=not args.no_controls)
        if not args.no_docx:
            from build_fallback_report import build
            build(args.output)
    else:
        run(model_path,args.labels,args.policy,folders,args.output,batch=args.batch,device=args.device)
        if not args.no_docx:
            from build_report import build
            build(args.output)
