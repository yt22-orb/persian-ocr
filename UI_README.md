# Persian Letter Scanner UI

The UI follows this flow:

1. Choose **Take a photo** or **Upload image**.
2. Capture/select one isolated Persian letter.
3. Review the full image and choose **Retake** or **Scan**.
4. The processing dialog stays visible while the model runs.
5. A successful scan shows the Persian letter, transliteration, crop-classifier confidence, four ranked model scores, and **Copy Text**. Model and recognition failures appear as retry/home dialogs.

## Run in VS Code

Open the repository folder in VS Code, then use its integrated terminal. Use Python 3.12 or 3.13 and install the UI dependencies:

```sh
python -m pip install -r requirements-ui.txt
```

Download `yolo11n.pt` from [release v1.0.2](https://github.com/yt22-orb/persian-ocr/releases/tag/v1.0.2) and put it in the repository root. The crop classifier checkpoint is already in `artifacts/crop_cnn/best.pt`. Start the server:

```sh
python serve_ui.py
```

Open `http://127.0.0.1:8501` if it does not open automatically. The endpoint uses the same confidence-gated YOLO and crop-classifier pipeline as the Colab notebook. CPU inference is supported; the first scan also loads both models and can take longer.

The server looks for the trained detector at the first matching location:

- the path in the `PERSIAN_OCR_MODEL` environment variable;
- `yolo11n.pt` in the repository root;
- `artifacts/yolo11n.pt`;
- `models/yolo11n.pt`.

The source archive does not contain `yolo11n.pt`. Until that published checkpoint is added locally, the interface shows the **Model can’t load** dialog after a scan.

## UI files

- `ui/index.html` — screens and dialogs
- `ui/styles.css` — reference palette, typography, responsive layout, and animations
- `ui/app.js` — camera/upload/preview/scan/result interactions
- `serve_ui.py` — static server and `/api/predict` model endpoint

Camera access works on localhost in modern browsers. Uploaded images are validated as PNG/JPEG and limited to 10 MB before inference.
