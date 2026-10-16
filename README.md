# Pothole Detection System — Streamlit

Streamlit migration of the Flask project. Existing Flask project remains unchanged.

## Features

- YOLO26 n/s/m/l and RT-DETR-L, confidence threshold 0.00–1.00
- Upload image and show bounding boxes + detection information
- Upload video (MP4/MOV/AVI/MKV/WEBM), sample at up to 10 FPS, export H.264 MP4
- Browser live camera via WebRTC with annotated frames (START/STOP)

**Difference from Flask:** Streamlit's native widgets and WebRTC START/STOP controls do not match the original HTML exactly. The camera's result is overlaid directly on the video rather than in the separate Detection Result panel. No standalone Run Detection control for camera. Public CPU hosting cannot guarantee real-time FPS.

## Local setup on Windows

1. Place this folder next to `runs2` in the original Pothole Detection project, naming it `streamlit_app`.
2. Open PowerShell in `streamlit_app` and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\copy_models.ps1
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Python 3.11 or 3.12 is recommended. If `py -3.12` is unavailable, install Python 3.12 or adjust the command to your installed compatible version. Ensure FFmpeg is on PATH for video processing.

3. Open the local URL (typically `http://localhost:8501`).
4. Test image first, then a short video, then browser camera.

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `inference.py`, `style.css`, `requirements.txt`, `packages.txt`, `.streamlit/config.toml`, and optional `assets/pothole-sign.png` to GitHub.
2. For initial CPU deployment, publish **only YOLO26n** checkpoint (`models/yolo26n.pt`) using Git LFS, if the hosting Git LFS checkout and memory limits allow it. Remove `models/*.pt` from `.gitignore` before staging the model and enable LFS tracking with `git lfs track 'models/*.pt'`. Large model files may be better stored in a Hugging Face model repository and downloaded at startup, but that requires a different model-loading configuration.
3. In Streamlit Community Cloud, create app from GitHub repo, set entrypoint `app.py` and choose Python 3.12 if offered.
4. Test CPU memory usage and image inference. Add other checkpoints one at a time only if RAM allows.

### Caveats

- Free hosting does not provide a GPU. YOLO26l and RT-DETR-L may exceed CPU RAM. This package preserves their selectors, but a checkpoint must be present to run them.
- Camera WebRTC depends on browser permissions, HTTPS, network/STUN connectivity, and can be unreliable behind restrictive NATs. A TURN server may be needed in some networks.
- The original Flask app allows 500 MB upload. This Streamlit version configures a 200 MB cap to reduce risk of memory exhaustion on free hosting. Raising the cap will not increase available RAM.
- Source video audio is not preserved, matching the original backend's FFmpeg `-an` behavior.
- Video upload bytes and final video bytes consume RAM. Prefer short videos on free hosting.
- Model loading uses a single-entry cache so multiple models are not retained simultaneously. CPU performance is unbenchmarked.
- The app has not been end-to-end tested with your actual checkpoints or on Community Cloud.
