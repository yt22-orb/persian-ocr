"""Models and image processing for bbox-cropped Persian glyph classification."""
import math

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn


def crop_and_clean(image, box, output_size=96, margin_ratio=0.06, background_tolerance=5):
    """Crop a glyph box, flatten its near-white background, and center it on white.

    ``box`` uses pixel coordinates ``[x1, y1, x2, y2]``. The small margin keeps
    detached dots and anti-aliased edges intact. Background is estimated from
    the crop border, so faint paper texture is removed without deleting
    disconnected components such as Persian dots.
    """
    gray = image.convert('L')
    x1, y1, x2, y2 = map(float, box)
    x1, y1 = math.floor(x1), math.floor(y1)
    x2, y2 = math.ceil(x2), math.ceil(y2)
    x1, x2 = max(0, x1), min(gray.width, x2)
    y1, y2 = max(0, y1), min(gray.height, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f'Invalid crop box: {box}')
    width, height = x2-x1, y2-y1
    margin = max(2, round(max(width, height)*margin_ratio))
    crop = gray.crop((max(0, x1-margin), max(0, y1-margin),
                      min(gray.width, x2+margin), min(gray.height, y2+margin)))
    pixels = np.asarray(crop, dtype=np.float32)
    border = max(1, min(margin, min(crop.size)//4))
    edge_pixels = np.concatenate((pixels[:border].ravel(), pixels[-border:].ravel(),
                                  pixels[:, :border].ravel(), pixels[:, -border:].ravel()))
    background = float(np.median(edge_pixels))
    if background < 80:
        raise ValueError('Crop border does not look like a light background.')
    normalized = np.clip(pixels*(255.0/background), 0, 255)
    normalized[(background-pixels) <= background_tolerance] = 255
    clean = Image.fromarray(np.rint(normalized).astype('uint8'), mode='L')

    # Reserve a fixed white border around the complete crop when resizing.
    inner = output_size-16
    scale = min(inner/clean.width, inner/clean.height)
    resized = clean.resize((max(1, round(clean.width*scale)),
                            max(1, round(clean.height*scale))), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (output_size, output_size), 255)
    canvas.paste(resized, ((output_size-resized.width)//2, (output_size-resized.height)//2))
    return canvas


def crop_tensor(image):
    pixels = np.asarray(image.convert('L'), dtype=np.float32)
    values = 1.0-2.0*(pixels/255.0)
    return torch.from_numpy(values.copy()).unsqueeze(0)


class CropLetterCNN(nn.Module):
    """Four-stage convolutional classifier for isolated, normalized glyph crops."""
    def __init__(self, dropout=0.25, classes=32):
        super().__init__()
        blocks = []
        channels = [1, 32, 64, 128, 192]
        for incoming, outgoing in zip(channels, channels[1:]):
            blocks.extend([
                nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                nn.BatchNorm2d(outgoing), nn.SiLU(inplace=True),
                nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                nn.BatchNorm2d(outgoing), nn.SiLU(inplace=True),
                nn.MaxPool2d(2),
            ])
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(192, classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def predict_crop_pipeline(detector, classifier, image, policy, device='cpu'):
    """Detect a glyph, clean its predicted box, then classify the crop."""
    from .inference import predict
    detected = predict(detector, image, policy['detector_policy'], device=device)
    if not detected['accepted']:
        return dict(accepted=False, reason=detected['reason'], detector=detected)
    box = detected['prediction']['box']
    crop = crop_and_clean(image, box)
    tensor = crop_tensor(crop).unsqueeze(0).to(device)
    with torch.inference_mode():
        scores = classifier(tensor).softmax(1)[0]
    confidence, class_id = scores.max(0)
    prediction = dict(class_id=int(class_id), confidence=float(confidence), box=box)
    top_scores, top_ids = scores.topk(min(4, scores.numel()))
    matches = [dict(class_id=int(index), confidence=float(score))
               for score, index in zip(top_scores, top_ids)]
    if prediction['confidence'] < policy['classifier_threshold']:
        return dict(accepted=False, reason='crop classifier low confidence', prediction=prediction,
                    matches=matches, crop=crop, detector=detected)
    return dict(accepted=True, reason='accepted', prediction=prediction,
                matches=matches, crop=crop, detector=detected)
