import json
import urllib.request
import base64
import io as _io
import time
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras

st.set_page_config(
    page_title="Medicinal Plant Classifier — Hamza Oulbayyad & Rida Bargach",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

  html, body, [class*="css"] {
      font-family: 'Plus Jakarta Sans', sans-serif;
  }
  .stApp {
      background: radial-gradient(circle at 50% -10%, #064e3b 0%, #0f172a 55%, #020617 100%);
      color: #f8fafc;
  }
  .hero-box {
      background: linear-gradient(135deg, rgba(6, 78, 59, 0.85), rgba(22, 163, 74, 0.75), rgba(16, 185, 129, 0.65));
      backdrop-filter: blur(12px);
      border: 1px solid rgba(74, 222, 128, 0.35);
      border-radius: 26px;
      padding: 36px 24px;
      text-align: center;
      color: white;
      margin-bottom: 28px;
      box-shadow: 0 20px 40px -15px rgba(16, 185, 129, 0.35);
  }
  .hero-badge {
      display: inline-block;
      background: rgba(255, 255, 255, 0.15);
      backdrop-filter: blur(6px);
      border: 1px solid rgba(255, 255, 255, 0.4);
      border-radius: 999px;
      padding: 8px 22px;
      font-size: 0.95rem;
      color: #ffffff;
      margin-top: 10px;
  }
  .model-card {
      background: rgba(30, 41, 59, 0.55);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 22px;
      padding: 24px 18px;
      text-align: center;
      height: 100%;
      transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
  }
  .model-card:hover {
      transform: translateY(-4px);
      border-color: rgba(74, 222, 128, 0.3);
      box-shadow: 0 12px 28px -8px rgba(0, 0, 0, 0.5);
  }
  .model-card-primary {
      background: linear-gradient(180deg, rgba(6, 78, 59, 0.45) 0%, rgba(15, 23, 42, 0.8) 100%);
      backdrop-filter: blur(12px);
      border: 1.5px solid #4ade80;
      border-radius: 22px;
      padding: 24px 18px;
      text-align: center;
      height: 100%;
      box-shadow: 0 10px 30px -10px rgba(74, 222, 128, 0.25);
      transition: transform 0.25s ease, box-shadow 0.25s ease;
  }
  .model-card-primary:hover {
      transform: translateY(-4px);
      box-shadow: 0 16px 36px -8px rgba(74, 222, 128, 0.35);
  }
  .primary-tag {
      display: inline-block;
      background: rgba(74, 222, 128, 0.18);
      color: #4ade80;
      border: 1px solid rgba(74, 222, 128, 0.4);
      font-size: 0.75rem;
      font-weight: 700;
      padding: 3px 12px;
      border-radius: 999px;
      margin-bottom: 10px;
      letter-spacing: 0.5px;
      text-transform: uppercase;
  }
  .secondary-tag {
      display: inline-block;
      background: rgba(148, 163, 184, 0.15);
      color: #94a3b8;
      font-size: 0.75rem;
      font-weight: 600;
      padding: 3px 12px;
      border-radius: 999px;
      margin-bottom: 10px;
  }
  .model-title {
      font-size: 1.3rem;
      font-weight: 800;
      color: #f8fafc;
      margin-bottom: 4px;
  }
  .model-sub {
      font-size: 0.85rem;
      color: #94a3b8;
      margin-bottom: 18px;
  }
  .bar-wrap { margin: 10px 0 6px 0; }
  .bar-label { font-size: 1.02rem; font-weight: 700; margin-bottom: 8px; }
  .bar-track {
      background: rgba(15, 23, 42, 0.8);
      border-radius: 999px;
      height: 24px;
      width: 100%;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.08);
      padding: 2px;
  }
  .bar-fill {
      height: 18px;
      border-radius: 999px;
      display: flex;
      align-items: center;
      padding-right: 10px;
      justify-content: flex-end;
      color: white;
      font-size: 0.82rem;
      font-weight: 800;
      min-width: 50px;
      transition: width 0.6s ease;
  }
  .green-fill { background: linear-gradient(90deg, #15803d, #22c55e, #4ade80); }
  .red-fill   { background: linear-gradient(90deg, #991b1b, #ef4444, #f87171); }
  .amber-fill { background: linear-gradient(90deg, #b45309, #f59e0b, #fbbf24); }
  .feedback-box {
      background: rgba(30, 41, 59, 0.6);
      border: 1px solid rgba(74, 222, 128, 0.25);
      border-radius: 18px;
      padding: 18px 22px;
      margin-top: 16px;
  }
  .stButton>button {
      border-radius: 14px !important;
      font-weight: 700 !important;
      background: rgba(30, 41, 59, 0.8) !important;
      border: 1px solid rgba(255, 255, 255, 0.15) !important;
      color: #ffffff !important;
      padding: 10px 18px !important;
      transition: all 0.25s ease !important;
  }
  .stButton>button:hover {
      background: linear-gradient(135deg, #16a34a, #15803d) !important;
      border-color: #4ade80 !important;
      transform: translateY(-2px) !important;
      box-shadow: 0 10px 20px -5px rgba(22, 163, 74, 0.45) !important;
  }
  .stTabs [data-baseweb="tab"] {
      background-color: rgba(30, 41, 59, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 8px 20px;
      color: #cbd5e1;
      font-weight: 600;
  }
  .stTabs [aria-selected="true"] {
      background-color: rgba(22, 163, 74, 0.45) !important;
      border-color: #4ade80 !important;
      color: #ffffff !important;
  }
</style>
""", unsafe_allow_html=True)

def save_correction_to_github(wrong: str, correct: str, img_bytes: bytes):
    token = st.secrets["GITHUB_TOKEN"]
    repo  = st.secrets["GITHUB_REPO"]

    img = Image.open(_io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    buf = _io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    fname = f"training_corrections/{int(time.time())}.jpg"

    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/contents/{fname}",
        data=json.dumps({"message": f"correction (Model 4): {wrong} -> {correct}",
                         "content": b64}).encode(),
        method="PUT",
        headers={"Authorization": f"token {token}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            img_url = json.loads(r.read())["content"]["html_url"]
    except Exception:
        img_url = None

    body = (f"**Model Evaluated:** Model 4 (Primary)\n\n"
            f"**Model 4 Prediction:** {wrong}\n\n"
            f"**User Correct Label:** {correct}\n\n"
            + (f"[📷 Plant Image]({img_url})\n\n" if img_url else "")
            + "*Submitted via Medicinal Plant Classifier*")
    req2 = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues",
        data=json.dumps({"title": f"Model 4 Correction: {wrong} -> {correct}",
                         "body": body,
                         "labels": ["correction"]}).encode(),
        method="POST",
        headers={"Authorization": f"token {token}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req2) as r:
        return json.loads(r.read())["html_url"]

@st.cache_resource
def load_models():
    m1 = keras.models.load_model("plant_model_v1.keras")
    m2 = keras.models.load_model("plant_model_v2.keras")
    m4 = keras.models.load_model("plant_model_v4.keras")
    return m1, m2, m4

m1, m2, m4 = load_models()

def predict(model, img):
    x = np.expand_dims(
        np.array(img.convert("RGB").resize((224, 224)), dtype="float32"), axis=0)
    p_non = float(model.predict(x, verbose=0)[0][0])
    return 1 - p_non

def label_v1v2(p_med):
    if p_med > 0.65:
        return "🟢 Medicinal", p_med, "green-fill", "#4ade80"
    if p_med < 0.35:
        return "🔴 Non-medicinal", 1 - p_med, "red-fill", "#f87171"
    return "🟡 Uncertain", max(p_med, 1 - p_med), "amber-fill", "#fbbf24"

def label_v4(p_med):
    if p_med >= 0.5:
        return "🟢 Usage médical important", p_med, "green-fill", "#4ade80"
    else:
        return "🔴 Usage médical quasi inexistant", 1 - p_med, "red-fill", "#f87171"

def bar(label, pct, fill_cls, text_color):
    return f"""
    <div class="bar-wrap">
      <div class="bar-label" style="color:{text_color};">{label}</div>
      <div class="bar-track">
        <div class="bar-fill {fill_cls}" style="width:{pct*100:.1f}%">
          {pct*100:.1f}%
        </div>
      </div>
    </div>"""

for k in ("results", "feedback_sent", "img_bytes"):
    if k not in st.session_state:
        st.session_state[k] = None

st.markdown("""
<div class="hero-box">
  <h1 style="margin:0 0 10px 0;font-size:2.3rem;font-weight:800;color:white;
             text-shadow:0 2px 10px rgba(0,0,0,0.25);">
    🌿 Medicinal Plant Classifier</h1>
  <p style="margin:0 0 14px 0;font-size:1.1rem;color:#ecfdf5;opacity:.95;">
    Compare three deep learning architectures — powered by <b>Model 4</b></p>
  <span class="hero-badge">
    ✨ Developed by: <b>Hamza Oulbayyad</b> &bull; <b>Rida Bargach</b>
  </span>
</div>
""", unsafe_allow_html=True)

col_results, col_input = st.columns([2.2, 1], gap="large")

with col_input:
    st.markdown("#### 📸 Input Plant Image")
    tab_up, tab_cam = st.tabs(["📁 Upload File", "📷 Take Photo"])
    with tab_up:
        uploaded = st.file_uploader("Upload plant image", type=["jpg", "jpeg", "png", "webp", "jfif"],
                                    label_visibility="collapsed")
    with tab_cam:
        camera = st.camera_input("Capture plant photo", label_visibility="collapsed")
    img_source = uploaded or camera

    if img_source:
        image = Image.open(img_source)
        image = ImageOps.exif_transpose(image)
        st.image(image, use_container_width=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    if st.button("🔬 Analyse Plant", type="primary", use_container_width=True):
        if img_source is None:
            st.warning("Please upload or capture a plant photo first.")
        else:
            with st.spinner("Running inference across all 3 models..."):
                image = Image.open(img_source)
                image = ImageOps.exif_transpose(image)
                p1 = predict(m1, image)
                p2 = predict(m2, image)
                p4 = predict(m4, image)
                st.session_state.results = (p1, p2, p4)
                st.session_state.feedback_sent = False
                buf = _io.BytesIO()
                image.convert("RGB").save(buf, format="JPEG")
                st.session_state.img_bytes = buf.getvalue()

with col_results:
    if st.session_state.results:
        p1, p2, p4 = st.session_state.results
        c1, c2, c3 = st.columns(3, gap="medium")

        models_info = [
            (c1, "Model 1", "Baseline Architecture", "Baseline v1", p1, label_v1v2, False),
            (c2, "Model 2", "Intermediate Architecture", "Intermediate v2", p2, label_v1v2, False),
            (c3, "Model 4", "Strong / Weak Medical Use", "⭐ Primary Model", p4, label_v4, True),
        ]

        for col, model_name, model_sub, badge_txt, p_med, lbl_fn, is_primary in models_info:
            lbl, conf, fill_cls, txt_col = lbl_fn(p_med)
            card_cls = "model-card-primary" if is_primary else "model-card"
            tag_cls = "primary-tag" if is_primary else "secondary-tag"
            with col:
                st.markdown(f"""
                <div class="{card_cls}">
                  <span class="{tag_cls}">{badge_txt}</span>
                  <div class="model-title">{model_name}</div>
                  <div class="model-sub">{model_sub}</div>
                  {bar(lbl, conf, fill_cls, txt_col)}
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

        with st.container():
            st.markdown("""
            <div class="feedback-box">
              <div style="font-size:1.05rem;font-weight:700;color:#4ade80;margin-bottom:4px;">
                🎯 Evaluate Primary Model (Model 4)
              </div>
              <div style="font-size:0.92rem;color:#cbd5e1;margin-bottom:12px;">
                Ignore Models 1 & 2 above — is <b>Model 4's</b> classification accurate for this plant?
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            if not st.session_state.feedback_sent:
                fc1, fc2 = st.columns(2, gap="medium")
                with fc1:
                    if st.button("✅ Yes, Model 4 is correct", use_container_width=True):
                        st.session_state.feedback_sent = "correct"
                        st.rerun()
                with fc2:
                    if st.button("❌ No, Model 4 is wrong", use_container_width=True):
                        st.session_state.feedback_sent = "wrong"
                        st.rerun()

            elif st.session_state.feedback_sent == "correct":
                st.success("Thank you for validating Model 4's prediction! 🎉")

            elif st.session_state.feedback_sent == "wrong":
                st.markdown("**Select the true classification for this plant:**")
                fix = st.selectbox("Correct label", ["", "Medicinal", "Non-medicinal"],
                                   label_visibility="collapsed")
                if fix and st.button("📨 Submit Correction for Model 4", use_container_width=True):
                    p1, p2, p4 = st.session_state.results
                    lbl4, _, _, _ = label_v4(p4)
                    with st.spinner("Saving correction to dataset..."):
                        try:
                            save_correction_to_github(
                                lbl4, fix, st.session_state.img_bytes)
                            st.session_state.feedback_sent = "done"
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error saving feedback: {e}")

            elif st.session_state.feedback_sent == "done":
                st.success("Correction saved! Thank you for helping improve Model 4 🎉")

    else:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    height:340px;background:rgba(30,41,59,0.3);border:1px dashed rgba(255,255,255,0.15);
                    border-radius:22px;color:#94a3b8;text-align:center;padding:20px;">
          <div style="font-size:2.5rem;margin-bottom:10px;">🌱</div>
          <div style="font-size:1.1rem;font-weight:600;color:#e2e8f0;">Ready to Analyse</div>
          <div style="font-size:0.95rem;margin-top:4px;">
            Upload a plant photo on the right and click <b>Analyse Plant</b>
          </div>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;font-size:.85rem;color:#94a3b8;margin-top:40px;
            border-top:1px solid rgba(255,255,255,0.08);padding-top:18px;">
  ⚠️ <b>Academic & Educational Project</b> — Do NOT use this tool for medical diagnosis or consumption decisions.<br>
  <b>Hamza Oulbayyad & Rida Bargach</b> • Dataset: Plants For A Future (pfaf.org)• <b>Version 1.1</b>
</div>
""", unsafe_allow_html=True)
