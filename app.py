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
/* ============================================================
   RADIO CARDS - Model and Variant (Streamlit nested label safe)
   ============================================================ */
.st-key-settings_panel [data-testid="stRadio"] div[role="radiogroup"] {
    display:flex!important;
    flex-direction:column!important;
    align-items:stretch!important;
    width:100%!important;
    gap:8px!important;
}
.st-key-settings_panel [data-testid="stRadio"] div[role="radiogroup"] > div {
    width:100%!important;
}
.st-key-settings_panel [data-testid="stRadio"] div[role="radiogroup"] label {
    box-sizing:border-box!important;
    display:flex!important;
    align-items:center!important;
    width:100%!important;
