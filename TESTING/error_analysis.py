"""Append exploratory error analysis calculated from completed prediction CSVs.

No checkpoint is loaded and no acceptance policy is changed. Standard-library
calculations keep the report builder usable with its existing dependencies.
"""
import csv
from collections import Counter
from pathlib import Path

ORDER=['DS1_Persian','DS1_Persian_Smoke','DS2_Persian_Hand','DS2_Persian_Hand_Smoke']
SHORT=['Printed original','Printed modified','Handwritten original','Handwritten modified']


def pct(value):
    return f'{value:.2f}%'


def rate(rows, field='correct'):
    return 100*sum(row[field]=='yes' for row in rows)/len(rows) if rows else 0.0


def read_rows(folder):
    data={}
    for name in ORDER:
        with (folder/name/'predictions.csv').open(encoding='utf-8-sig',newline='') as handle:
            data[name]=list(csv.DictReader(handle))
    return data


def match(original, modified):
    by_source={row['source_relative_path']:row for row in original}
    if len(by_source)!=len(original) or len(original)!=len(modified):
        raise ValueError('Original and modified identities must be unique and complete.')
    pairs=[]
    seen=set()
    for row in modified:
        source=row['source_relative_path']
        before=by_source[source]
        if source in seen or before['image_file_sha256']!=row['source_file_sha256'] or before['expected_label_id']!=row['expected_label_id']:
            raise ValueError('Original and modified image identity mismatch.')
        seen.add(source)
        pairs.append(dict(row,correct_original=before['correct']))
    if seen!=set(by_source):
        raise ValueError('Missing original and modified image pairs.')
    return pairs


def choose(rows,modification,low=None,high=None):
    values=[row for row in rows if row['modification']==modification]
    if low is None:
        return values
    return [row for row in values if (abs(float(row['rotation_angle_degrees']))>low or
            (low==5 and abs(float(row['rotation_angle_degrees']))==5)) and
            abs(float(row['rotation_angle_degrees']))<=high]


def transitions(rows):
    lost=sum(row['correct_original']=='yes' and row['correct']=='no' for row in rows)
    gained=sum(row['correct_original']=='no' and row['correct']=='yes' for row in rows)
    return lost,gained


def append_error_analysis(doc,folder,table):
    data=read_rows(Path(folder))
    printed=match(data[ORDER[0]],data[ORDER[1]])
    hand=match(data[ORDER[2]],data[ORDER[3]])
    total=sum(len(rows) for rows in data.values())

    # Page 7: outcome definitions and evidence for the training mismatch.
    doc.add_page_break()
    doc.add_heading('Detection and prediction error analysis',1)
    doc.add_paragraph('The strongest supported explanations are a mismatch with natural handwriting and unfamiliar fonts, '
                      'limited rotation tolerance, and difficulty distinguishing dots and small strokes. Blur and brightness '
                      'or contrast changes added little aggregate error within the tested ranges. These are inferences from '
                      'the recorded results; the underlying mechanisms have not been established by controlled experiments.')
    doc.add_heading('Analysis method',1)
    doc.add_paragraph(f'This analysis uses all {total:,} recorded image outcomes and {len(printed)+len(hand):,} verified '
                      'original and modified pairs. Source paths, file hashes, and expected labels must agree before pairing. '
                      'The model and original 80% acceptance rule remain fixed. Accuracy includes rejected recognitions as incorrect.')
    rows=[]
    for name,label in zip(ORDER,SHORT):
        values=data[name]; counts=Counter(row['status'] for row in values); n=len(values)
        wrong=sum(row['status']=='accepted' and row['correct']=='no' for row in values)
        rows.append([label,pct(100*counts['no_detection']/n),pct(100*counts['low_confidence']/n),
                     pct(100*counts['multiple_detections']/n),pct(100*wrong/n)])
    table(doc,['Dataset','No candidate at 3%','Low confidence','Multiple candidates','Accepted wrong letter'],rows,[1.8,1.35,1.35,1.35,1.35])
    doc.add_paragraph('Every percentage in this table uses all images in its dataset. No candidate means that no detection '
                      'survived the saved 3% floor and inference settings. Low confidence means exactly one candidate was '
                      'present but below 80%. The last column contains confidently wrong letter assignments.')
    doc.add_paragraph('For handwriting, obtaining a usable candidate is the largest bottleneck. For modified print, '
                      'candidates usually remain available, but confidence falls and wrong letter assignments increase. '
                      'The CSVs have no ground truth bounding boxes, so these outcomes cannot establish localization IoU or detection mAP.')
    doc.add_heading('Training context',1)
    doc.add_paragraph('The tested checkpoint SHA256 matches artifacts/yolo_metadata.json in the repository. Its documented '
                      'training used rendered font images, rotations up to 3 degrees, and no horizontal or vertical flips. '
                      'Training fonts were Lateef, Markazi Text, Vazirmatn, Gulzar, Noto Naskh Arabic, and Noto Sans Arabic. '
                      'This supports a training distribution mismatch for natural handwriting and severe transformations. '
                      'It does not establish that any one image property caused an individual failure.')

    # Page 8: same-image rotation comparisons and direction associations.
    doc.add_page_break()
    doc.add_heading('Rotation sensitivity',1)
    doc.add_paragraph('Printed accuracy has a clear deterioration region around 20 degrees. This table includes rotation '
                      'only, without flips. Original accuracy refers to the exact printed images assigned to each angle band.')
    rows=[]; hand_counts=[]
    for low,high in [(5,10),(10,15),(15,20),(20,25),(25,30),(30,35),(35,40)]:
        p=choose(printed,'rotation',low,high); h=choose(hand,'rotation',low,high); hand_counts.append(len(h))
        rows.append([f'{"" if low==5 else ">"}{low} to {high}',f'{len(p):,}',pct(rate(p,'correct_original')),pct(rate(p)),pct(rate(h))])
    table(doc,['Absolute angle degrees','Printed images','Original print accuracy','Rotated print accuracy','Rotated hand accuracy'],rows,[1.35,1.0,1.6,1.6,1.65])
    doc.add_paragraph(f'Handwritten bands contain {min(hand_counts):,} to {max(hand_counts):,} images each. '
                      'Above 20 degrees is an observed warning region for print, and above 30 degrees produces severe '
                      'degradation. This is not a safe cutoff for every letter; some classes fail substantially earlier.')
    severe=choose(printed,'rotation',35,40)
    low=sum(row['status']=='low_confidence' for row in severe)
    wrong=sum(row['status']=='accepted' and row['correct']=='no' for row in severe)
    doc.add_paragraph(f'At 35 to 40 degrees, {pct(100*low/len(severe))} of printed images were rejected for low confidence, '
                      f'while {pct(100*wrong/len(severe))} received an accepted wrong prediction.')
    doc.add_heading('Rotation direction',1)
    rows=[]
    for values,label in [(printed,'Printed'),(hand,'Handwritten')]:
        rotated=choose(values,'rotation')
        clockwise=[row for row in rotated if float(row['rotation_angle_degrees'])<0]
        counter=[row for row in rotated if float(row['rotation_angle_degrees'])>0]
        rows.append([label,pct(rate(clockwise)),pct(rate(counter))])
    table(doc,['Dataset','Clockwise accuracy','Counterclockwise accuracy'],rows,[2.4,2.4,2.4])
    doc.add_paragraph('The average printed angle magnitudes were almost identical at approximately 22.3 degrees. '
                      'The direction difference could involve glyph slant or preprocessing geometry; its cause is unconfirmed.')
    doc.add_paragraph('Small rotations sometimes improved results, especially for handwriting. The transform also resizes '
                      'the expanded image and changes its framing, so the improvement cannot be attributed solely to '
                      'correcting orientation. A zero degree control with the same resampling and padding would help separate these effects.')

    # Page 9: flipping and photometric transforms.
    doc.add_page_break()
    doc.add_heading('Flip blur and brightness sensitivity',1)
    doc.add_paragraph('Both flip directions are destructive in these tests, with vertical flips worse for print. '
                      'The following comparisons use the same original and flipped images.')
    rows=[]
    for values,label in [(printed,'Printed'),(hand,'Handwritten')]:
        for direction in ['horizontal','vertical']:
            g=[row for row in choose(values,'flip') if row['flip_type']==direction]
            rows.append([label+' '+direction,f'{len(g):,}',pct(rate(g,'correct_original')),pct(rate(g))])
    table(doc,['Dataset and flip','Images','Original accuracy','Flipped accuracy'],rows,[2.5,.9,1.9,1.9])
    doc.add_paragraph(f'Combining rotation and flipping produced {pct(rate(choose(printed,"rotationFlip")))} printed '
                      f'accuracy and {pct(rate(choose(hand,"rotationFlip")))} handwritten accuracy. A flip can change a '
                      'letter\'s defining orientation or dot position, producing a mirrored symbol that no longer represents '
                      'its original class. These scores measure preservation of the original label under mirroring. '
                      'Flips are useful stress tests, but require care as training augmentation.')
    doc.add_heading('Photometric changes in printed images',1)
    blur=choose(printed,'blur'); tone=choose(printed,'brightnessContrast')
    table(doc,['Transformation','Tested range','Images','Original accuracy','Modified accuracy'],[
        ['Gaussian blur','Radius 0.6 to 1.3 pixels',len(blur),pct(rate(blur,'correct_original')),pct(rate(blur))],
        ['Brightness and contrast','Each factor 0.75 to 0.90 or 1.10 to 1.25',len(tone),pct(rate(tone,'correct_original')),pct(rate(tone))],
    ],[1.4,2.0,.8,1.5,1.5])
    lost,gained=transitions(blur)
    doc.add_paragraph(f'With blur, {lost} previously correct images became incorrect and {gained} previously incorrect '
                      'images became correct. Increasing blur within this range did not produce a consistent decline. '
                      'No reliable blur failure threshold was identified, and stronger blur was not tested.')
    lost,gained=transitions(tone)
    doc.add_paragraph(f'For brightness and contrast, only {lost+gained} images changed correctness: {gained} improved and '
                      f'{lost} deteriorated. All four brighter or darker and higher or lower contrast combinations had '
                      'zero net accuracy change in this run. The factors changed together, preventing clean identification '
                      'of their independent thresholds. These two transforms were not applied to handwriting.')

    # Page 10: letters and fonts with explicit sample sizes.
    doc.add_page_break()
    doc.add_heading('Letter and font effects',1)
    doc.add_paragraph('No single rotation threshold applies to every letter. The strongest printed examples below '
                      'compare each transformed subset with its own originals.')
    rows=[]
    for letter,limit in [('alef',15),('beh',25),('kaf',30)]:
        g=[row for row in choose(printed,'rotation') if row['expected_label']==letter and abs(float(row['rotation_angle_degrees']))>limit]
        rows.append([letter,f'Above {limit}',len(g),pct(rate(g,'correct_original')),pct(rate(g))])
    table(doc,['Letter','Angle degrees','Images','Original accuracy','Rotated accuracy'],rows,[1.0,1.5,.8,1.95,1.95])
    alef=[row for row in choose(printed,'rotation',10,15) if row['expected_label']=='alef']
    heh=[row for row in choose(printed,'rotation',35,40) if row['expected_label']=='heh']
    doc.add_paragraph(f'Alef already fell to {pct(rate(alef))} at 10 to 15 degrees across {len(alef)} images. '
                      f'Printed heh retained {pct(rate(heh))} at 35 to 40 degrees across {len(heh)} images. '
                      'These differences suggest sensitivity to particular stroke shapes. The small letter and angle '
                      'groups are diagnostic findings rather than precise universal cutoffs.')
    doc.add_heading('Original printed fonts',1)
    fonts={}
    for row in data[ORDER[0]]:
        fonts.setdefault(row['source_relative_path'].split('/')[0],[]).append(row)
    rows=[]
    for font in ['Samim','Parastoo-Print','B Badkonak','B Masjed','B Kaj','B Tir']:
        if font in fonts:
            g=fonts[font]
            rows.append([font,f'{sum(row["correct"]=="yes" for row in g)}/{len(g)}',pct(rate(g))])
    table(doc,['Font','Correct images','Accuracy'],rows,[3.2,2.0,2.0])
    high=sum(rate(g)>=90 for g in fonts.values()); low=sum(rate(g)<=10 for g in fonts.values())
    doc.add_paragraph(f'Of {len(fonts)} font folders, {high} reached at least 90% accuracy and {low} achieved 10% or less. '
                      'Sample inspection of poorly performing fonts showed strongly stylized or geometric glyphs, '
                      'consistent with difficulty transferring from the training fonts. There is one sample per '
                      'letter per font, and some font variants contain identical images, so these rankings are descriptive.')

    # Page 11: dot confusions and counterfactual confidence gates.
    doc.add_page_break()
    doc.add_heading('Handwritten letter confusions and confidence',1)
    values=data[ORDER[2]]
    wrong=[row for row in values if row['status']=='accepted' and row['correct']=='no']
    counts=Counter((row['expected_label'],row['predicted_label']) for row in wrong)
    dot_pairs=[('kheh','heh-jimi'),('zal','dal'),('shin','sin'),('jim','heh-jimi'),('gheyn','eyn'),('cheh','heh-jimi'),('za','ta')]
    table(doc,['Expected letter','Predicted letter','Accepted errors'],[[a,b,counts[a,b]] for a,b in dot_pairs[:5]],[2.4,2.4,2.4])
    dot_count=sum(counts[pair] for pair in dot_pairs)
    doc.add_paragraph(f'Including cheh to heh-jimi and za to ta, these seven dot losing confusions account for '
                      f'{dot_count} of {len(wrong)} confidently wrong handwritten predictions '
                      f'({pct(100*dot_count/len(wrong))}). This is consistent with recognizing the main letter body '
                      'while missing or discounting dots. The CSVs cannot distinguish resolution, faint strokes, '
                      'handwriting variation, resizing, or bounding box placement as the underlying cause. '
                      'The 77 by 95 pixel source images make small feature loss a plausible contributor.')
    grouped={}
    for row in values:
        grouped.setdefault(row['expected_label'],[]).append(row)
    zero=sorted(letter for letter,rows in grouped.items() if rate(rows)==0)
    best=sorted(grouped,key=lambda letter:rate(grouped[letter]),reverse=True)[:2]
    doc.add_paragraph('Classes with zero correct accepted predictions were '+', '.join(zero)+'. '
                      'Each had 1,000 images. The strongest handwritten classes were '+
                      ' and '.join(letter+' at '+pct(rate(grouped[letter])) for letter in best)+
                      ', which still indicates poor overall handwriting coverage.')
    doc.add_heading('Diagnostic confidence thresholds',1)
    rows=[]
    for threshold in [80,50,3]:
        selected=[row for row in values if int(row['candidate_count'])==1 and float(row['top_confidence_percent'] or 0)>=threshold]
        correct=sum(row['top_candidate_label']==row['expected_label'] for row in selected)
        rows.append([f'{threshold}%',pct(100*correct/len(values)),pct(100*correct/len(selected)) if selected else 'Unavailable'])
    table(doc,['Acceptance threshold','Overall accuracy','Correct among accepted'],rows,[2.4,2.4,2.4])
    doc.add_paragraph('These retrospective calculations preserve the single candidate rule and do not change the '
                      'recorded 80% test policy. Lowering the gate cannot solve the underlying problem: many images '
                      'have no candidate and many candidate classes are wrong. Operational thresholds should be '
                      'selected using separate validation data.')

    # Page 12: conclusions, limits and traceable source references.
    doc.add_page_break()
    doc.add_heading('Interpretation and improvement priorities',1)
    for title,text in [
        ('Representative handwriting and fonts','The largest supported mismatch is the transfer from rendered training fonts to natural handwriting and difficult glyph styles. Training or adaptation with representative samples should be evaluated using separate validation and test groups.'),
        ('Preserving dots and small strokes','Compare image normalization, margins, and resizing methods that retain detached dots. The observed dot confusions support this investigation, but do not establish which preprocessing method will help.'),
        ('Rotation handling','Investigate deskewing and broader training rotation coverage, particularly for alef, beh, and kaf. The observed warning region around 20 degrees is not a guarantee that lower angles are safe for every letter.'),
        ('Confidence calibration','Evaluate acceptance thresholds on separate validation data, reporting both correctly recognized inputs and confidently wrong predictions. Increasing the number of accepted inputs alone is not evidence of better recognition.'),
    ]:
        doc.add_heading(title,1)
        doc.add_paragraph(text)
    doc.add_heading('Limits of the threshold findings',1)
    doc.add_paragraph('Each image received only one modification. The reported ranges are exploratory associations, '
                      'not exact experimentally established cutoffs. Precise thresholds require the same images '
                      'tested at fixed strengths, including an unchanged control. Brightness and contrast should '
                      'be varied separately. Font variants with identical images should not be treated as independent font families.')
    doc.add_paragraph('This analysis concerns the standalone YOLO checkpoint used for the recorded tests. It does not '
                      'measure the repository\'s separate crop classifier. Proposed improvements have not been '
                      'implemented or tested here, and the analysis does not establish unseen writer or font generalization.')
    doc.add_heading('Sources',1)
    doc.add_paragraph('Recorded evidence: Results contains the four predictions.csv files, report.json files, and paired_comparison.csv. '
                      'Dataset sources and transformations are documented in Test Set ReadMe.md. Training provenance is '
                      'recorded in README.md and artifacts/yolo_metadata.json at repository revision 6a48739.')
    doc.add_paragraph('Repository and training record: https://github.com/yt22-orb/persian-ocr/tree/6a48739ab47bd5149d243b9001eca88a2ac302db')
