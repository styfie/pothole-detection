"""Pothole Detection System — self-contained Streamlit interface.

Keep inference.py and model checkpoints in the same locations as before.
No external style.css is required.
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from inference import MODELS, decode_image, device_label, load_model, predict, process_video

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Pothole Detection System", page_icon="🛣️", layout="wide", initial_sidebar_state="collapsed")

# Styling lives in this file so an outdated/misplaced style.css cannot override it.
CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {--ink:#1d2230;--text:#252936;--muted:#77776f;--yellow:#f0df7b;--pale:#fff9dc;--box:#f0f0ed;}
html,body,[class*="stApp"]{font-family:Inter,Arial,sans-serif!important;color:var(--text)}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#f7f6f1!important}
.block-container{max-width:100%!important;padding:.6rem 1.3rem 2rem!important}
header[data-testid="stHeader"]{background:transparent!important}
.app-header{display:flex;align-items:center;gap:15px;min-height:92px;margin:-10px -21px 8px;padding:8px 26px;background:linear-gradient(105deg,#ffffc5 0%,#f7edaa 52%,#f0df7b 100%);border-bottom:1px solid #e5d46e}
.brand-logo{width:90px;height:74px;object-fit:contain}.brand-fallback{font-size:45px}
.brand-title{color:var(--ink);font-size:30px;font-weight:800;letter-spacing:-.7px;line-height:1.1}
.brand-subtitle{margin-top:7px;color:#5f604f;font-size:12px;font-weight:500}
.section-heading{font-size:19px;font-weight:800;color:var(--ink);margin:0 0 15px;letter-spacing:-.35px}
/* Keyed st.container elements are the panels themselves, regardless of the wrapper test-id. */
.st-key-settings_panel,.st-key-workspace_panel{background-color:#fff!important;background-image:none!important;border:1px solid #deded8!important;border-radius:12px!important;box-shadow:0 3px 13px rgba(35,35,30,.045)!important}
.st-key-settings_panel>div,.st-key-workspace_panel>div{background-color:transparent!important;background-image:none!important}
.stRadio label,.stNumberInput label,.stFileUploader label{font-size:12px!important;font-weight:750!important;color:var(--ink)!important}
[data-testid="stRadio"] [role="radiogroup"]{gap:7px}
[data-testid="stRadio"] label:has(input:checked){background:var(--pale);border-radius:8px}
/* Border the number input control itself, not its entire labeled field. */
.st-key-settings_panel [data-testid="stNumberInput"] [data-baseweb="input"],
.st-key-settings_panel [data-testid="stNumberInput"] [data-baseweb="base-input"]{background:var(--pale)!important;border:1px solid #a9a38e!important;border-radius:8px!important;box-shadow:none!important;overflow:hidden}
.st-key-settings_panel [data-testid="stNumberInput"] input,
.st-key-settings_panel [data-testid="stNumberInput"] button{background:var(--pale)!important;color:var(--ink)!important}
.st-key-settings_panel [data-testid="stNumberInput"] [data-baseweb="input"]:focus-within{border-color:#b6a23c!important;box-shadow:0 0 0 2px rgba(240,223,123,.25)!important}
.stButton button[kind="primary"],button[data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,#ffffc5,#f0df7b)!important;color:var(--ink)!important;border:1px solid #d7c45e!important;border-radius:9px!important;font-size:12px!important;font-weight:800!important;min-height:46px}
.stButton button[kind="primary"]:hover{background:linear-gradient(135deg,#fffdb6,#e8d66f)!important;color:var(--ink)!important}
[data-testid="stFileUploaderDropzone"]{background:var(--box)!important;background-image:none!important;border:1px dashed #d8d6cd!important;border-radius:11px!important;min-height:0!important;padding:7px 12px!important}
[data-testid="stFileUploaderDropzone"] button{background:#ffffc5!important;color:var(--ink)!important;border:1px solid #d8c862!important}
.empty-result{min-height:160px;display:flex;align-items:center;justify-content:center;text-align:center;border:1px dashed #dedcd2;border-radius:11px;background:var(--box)!important;color:#85847d;font-size:12px;padding:25px}
.stat-box{background:var(--box);border:1px solid #e9e6d9;border-radius:9px;padding:10px 8px;min-height:70px}
.stat-label{font-size:10px;font-weight:700;color:var(--muted)}.stat-value{margin-top:7px;font-size:12px;font-weight:800;color:var(--ink);overflow-wrap:anywhere}
.camera-intro{min-height:95px;text-align:center;padding:16px;background:var(--box);border:1px dashed #d8d6cd;border-radius:9px}.camera-intro strong{font-size:15px}.camera-intro p{font-size:11px;color:#7c7f87;margin:8px 0}
/* Compact native uploader: keep Streamlit file selection and removal functional. */
.st-key-workspace_panel [data-testid="stFileUploaderDropzone"]{min-height:0!important;padding:7px 12px!important}
.st-key-workspace_panel [data-testid="stFileUploaderDropzoneInstructions"]{display:none!important}
.st-key-workspace_panel [data-testid="stFileUploaderDropzone"] button{min-height:32px!important;padding:4px 12px!important}
.st-key-workspace_panel [data-testid="stFileUploaderFile"]{padding:5px 10px!important;margin-top:4px!important;background:#f0f0ed!important;border-radius:8px!important}
.st-key-workspace_panel [data-testid="stFileUploader"]{margin-bottom:2px!important}
/* Keep the settings panel visually balanced against the right workspace. */
.st-key-settings_panel{min-height:730px!important}
/* Comfortable bottom breathing room for the result box. */
.st-key-workspace_panel{padding-bottom:24px!important;box-sizing:border-box!important}

/* Model and variant choices: bordered rounded cards with hover feedback. */
.st-key-settings_panel [data-testid="stRadio"] [role="radiogroup"]{display:flex!important;flex-direction:column!important;gap:8px!important}
.st-key-settings_panel [data-testid="stRadio"] [role="radiogroup"]>label{width:100%!important;min-height:45px!important;padding:9px 12px!important;background:#fff!important;border:1px solid #dedcd5!important;border-radius:10px!important;cursor:pointer!important;transition:background-color .18s,border-color .18s!important}
.st-key-settings_panel [data-testid="stRadio"] [role="radiogroup"]>label:hover{background:#fffbe8!important;border-color:#e5d46e!important}
.st-key-settings_panel [data-testid="stRadio"] [role="radiogroup"]>label:has(input:checked){background:#fff7b8!important;border-color:#e3cb55!important}
/* Three equal-width input-source segments. */
.st-key-workspace_panel [data-testid="stSegmentedControl"]{width:100%!important}
.st-key-workspace_panel [data-testid="stSegmentedControl"] [role="radiogroup"]{display:flex!important;width:100%!important;gap:0!important}
.st-key-workspace_panel [data-testid="stSegmentedControl"] button{flex:1 1 0!important;min-height:48px!important;border:1px solid #e4dfcf!important;border-radius:0!important;background:#faf9f5!important;color:#555!important;transition:background-color .18s!important}
.st-key-workspace_panel [data-testid="stSegmentedControl"] button:hover{background:#fffbe8!important}
.st-key-workspace_panel [data-testid="stSegmentedControl"] button[aria-checked="true"],.st-key-workspace_panel [data-testid="stSegmentedControl"] button[aria-pressed="true"],.st-key-workspace_panel [data-testid="stSegmentedControl"] button[data-state="active"]{background:#fff4a8!important;color:#1d2230!important}
.st-key-workspace_panel [data-testid="stSegmentedControl"] button:first-child{border-radius:10px 0 0 10px!important}
.st-key-workspace_panel [data-testid="stSegmentedControl"] button:last-child{border-radius:0 10px 10px 0!important}
/* Dedicated nested media surfaces: gray backdrop, centered 70%-width media. */
.st-key-input_preview,.st-key-output_preview{background:#f0f0ed!important;background-image:none!important;border:1px solid #dedcd5!important;border-radius:12px!important;padding:16px!important;margin-top:10px!important;margin-bottom:18px!important;box-sizing:border-box!important}
.st-key-input_preview>div,.st-key-output_preview>div{background:transparent!important}
.st-key-input_preview img,.st-key-output_preview img,.st-key-input_preview video,.st-key-output_preview video{border-radius:7px!important}
.st-key-workspace_panel{padding-bottom:24px!important}

/* Preview size is controlled by dedicated columns in Python. */
@media(max-width:850px){.st-key-settings_panel{min-height:0!important}}
@media(max-width:850px){.brand-title{font-size:22px}.brand-subtitle{font-size:10px}.app-header{padding:10px 14px}.brand-logo{width:60px;height:55px}.block-container{padding:.7rem!important}}
"""
st.html(f"<style>{CSS}</style>")


def reset_result() -> None:
    st.session_state.result = None


def header() -> None:
    logo = ROOT / "assets" / "pothole-sign.png"
    if logo.is_file():
        encoded = base64.b64encode(logo.read_bytes()).decode("ascii")
        logo_html = f'<img class="brand-logo" src="data:image/png;base64,{encoded}" alt="Pothole sign">'
    else:
        logo_html = '<span class="brand-fallback">⚠</span>'
    st.markdown(
        f'<div class="app-header">{logo_html}<div>'
        '<div class="brand-title">Pothole Detection System</div>'
        '<div class="brand-subtitle">Detect potholes from images, videos, and live camera input using YOLO26 and RT-DETR.</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def show_result(result: dict | None) -> None:
    st.markdown('<div class="section-heading">Detection Result</div>', unsafe_allow_html=True)
    if result is None:
        st.markdown('<div class="empty-result">Detection results will appear here after running the model.</div>', unsafe_allow_html=True)
        return
    with st.container(border=False, key="output_preview"):
        preview_left, preview_center, preview_right = st.columns([1.5, 7, 1.5], gap="small")
        with preview_center:
            if result["kind"] == "image":
                st.image(result["content"], use_container_width=True)
            else:
                st.video(result["content"])
    if result["kind"] == "video":
        st.download_button("Download Result Video", result["content"], file_name="pothole_detection_result.mp4", mime="video/mp4", use_container_width=True)
    stats = result["stats"]
    detection_label = f'{stats["detections"]} frame detections' if result["kind"] == "video" else f'{stats["detections"]} potholes'
    values = [
        ("Detections", detection_label),
        ("Model", result["model"]),
        ("Confidence", f'{result["confidence"]:.2f}'),
        ("Inference Time", f'{stats["inference_time_ms"]:.1f} ms' if stats.get("inference_time_ms") is not None else "-"),
        ("Device", stats["device"]),
    ]
    for column, (label, value) in zip(st.columns(len(values), gap="small"), values):
        with column:
            st.markdown(
                f'<div class="stat-box"><div class="stat-label">{html.escape(label)}</div>'
                f'<div class="stat-value">{html.escape(str(value))}</div></div>',
                unsafe_allow_html=True,
            )


if "result" not in st.session_state:
    st.session_state.result = None

header()
left, right = st.columns([360, 810], gap="small")

with left:
    with st.container(border=True, key="settings_panel"):
        st.markdown('<div class="section-heading">☷ &nbsp; Model Settings</div>', unsafe_allow_html=True)
        model_family = st.radio("Model", ["YOLO26", "RT-DETR"], key="model_family", on_change=reset_result)
        variants = ["Nano (n)", "Small (s)", "Medium (m)", "Large (l)"] if model_family == "YOLO26" else ["Large (L)"]
        # Streamlit can retain an old radio choice when the family changes.
        if st.session_state.get("model_variant") not in variants:
            st.session_state["model_variant"] = variants[0]
        variant = st.radio("Variant", variants, key="model_variant", on_change=reset_result)
        confidence = st.number_input("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01, format="%.2f", key="confidence", on_change=reset_result)
        st.caption("Value between 0.00 and 1.00")
        run_placeholder = st.empty()

with right:
    with st.container(border=True, key="workspace_panel"):
        st.markdown('<div class="section-heading">⇧ &nbsp; Input Source</div>', unsafe_allow_html=True)
        source = st.segmented_control("Input Source", options=["Image", "Video", "Camera"], default="Image", selection_mode="single", label_visibility="collapsed", key="source", on_change=reset_result)
        upload = None
        if source == "Image":
            upload = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], help="Drag and drop an image here, or choose a file from your computer.", key="image_upload", on_change=reset_result)
            if upload is not None:
                with st.container(border=False, key="input_preview"):
                    preview_left, preview_center, preview_right = st.columns([1.5, 7, 1.5], gap="small")
                    with preview_center:
                        st.image(upload, use_container_width=True)
        elif source == "Video":
            upload = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv", "webm"], help="MP4, MOV, AVI, MKV, WEBM", key="video_upload", on_change=reset_result)
            if upload is not None:
                with st.container(border=False, key="input_preview"):
                    preview_left, preview_center, preview_right = st.columns([1.5, 7, 1.5], gap="small")
                    with preview_center:
                        st.video(upload)
        else:
            st.markdown('<div class="camera-intro"><strong>Browser Camera</strong><p>Open your browser camera for real-time pothole detection.</p></div>', unsafe_allow_html=True)
            st.caption("Camera permission will be requested by your browser")
            st.caption("Press START to open the camera; STOP to close it. Live inference starts with the camera.")
            try:
                live_model = load_model(model_family, variant)
            except Exception as exc:
                st.warning(str(exc))
                live_model = None
            if live_model is not None:
                def video_callback(frame: av.VideoFrame) -> av.VideoFrame:
                    image = frame.to_ndarray(format="bgr24")
                    annotated, _, _ = predict(image, live_model, float(confidence), imgsz=640)
                    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

                webrtc_streamer(
                    key=f"camera_{model_family}_{variant}_{confidence}",
                    mode=WebRtcMode.SENDRECV,
                    video_frame_callback=video_callback,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                )
                st.caption("Camera: bounding boxes are shown on the live video. CPU hosting may have low FPS.")
        if source != "Camera":
            show_result(st.session_state.result)

with left:
    if source == "Camera":
        run_placeholder.info("For Camera, use START / STOP in the camera panel.")
    else:
        run = run_placeholder.button("▶  Run Detection", use_container_width=True, type="primary")
        if run:
            if upload is None:
                st.warning(f"Please select an {source.lower()} first.")
            else:
                try:
                    model_label, _ = MODELS[(model_family, variant)]
                    with st.spinner(f"Loading {model_label}..."):
                        model = load_model(model_family, variant)
                    if source == "Image":
                        with st.spinner("Detecting potholes..."):
                            frame = decode_image(upload.getvalue())
                            annotated, count, ms = predict(frame, model, float(confidence))
                            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            st.session_state.result = {
                                "kind": "image", "content": rgb, "model": model_label,
                                "confidence": float(confidence),
                                "stats": {"detections": count, "inference_time_ms": ms, "device": device_label()},
                            }
                    else:
                        progress = st.progress(0.0, text="Processing video...")
                        with st.spinner("Processing video frames and encoding MP4..."):
                            data, stats = process_video(
                                upload.getvalue(), Path(upload.name).suffix, model,
                                float(confidence), progress_callback=lambda value: progress.progress(value),
                            )
                            st.session_state.result = {
                                "kind": "video", "content": data, "model": model_label,
                                "confidence": float(confidence), "stats": stats,
                            }
                        progress.empty()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Detection failed: {exc}")
