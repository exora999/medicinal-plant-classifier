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
    page_title="Medicinal Plant Classifier — Hamza Oulbayyad",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
  .stApp {
      background: radial-gradient(circle at 50% 0%, #052e16 0%, #0f172a 60%, #020617 100%);
      color: #f3f4f6;
  }
  .model-card {
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 18px;
      padding: 20px 16px;
      text-align: center;
      height: 100%;
  }
  .model-title {
      font-size: 1.2rem; font-weight: 700;
      color: #4ade80; margin-bottom: 4px;
  }
  .model-sub {
      font-size: .88rem; color: #9ca3af; margin-bottom: 14px;
  }
  .bar-wrap{margin:8px 0 12px 0;}
  .bar-label{font-size:1rem;font-weight:600;margin-bottom:5px;color:#4ade80;}
  .bar-track{background:rgba(255,255,255,0.08);border-radius:999px;height:22px;
             width:100%;overflow:hidden;border:1px solid rgba(255,255,255,0.05);}
  .bar-fill{height:22px;border-radius:999px;display:flex;align-items:center;
            padding-right:10px;justify-content:flex-end;color:white;
            font-size:.85rem;font-weight:700;min-width:48px;}
  .green-fill{background:linear-gradient(90deg,#14532d,#16a34a,#4ade80);}
  .gray-fill {background:linear-gradient(90deg,#374151,#4b5563,#9ca3af);}
  .stButton>button{
      border-radius:14px!important;font-weight:700!important;
      background:rgba(255,255,255,0.05)!important;
      border:1px solid rgba(255,255,255,0.15)!important;
      color:#ffffff!important;transition:all 0.3s ease!important;
  }
  .stButton>button:hover{
      background:linear-gradient(135deg,#16a34a,#14532d)!important;
      border-color:#4ade80!important;transform:translateY(-2px)!important;
      box-shadow:0 10px 20px -3px rgba(22,163,74,0.4)!important;
  }
  .stTabs [data-baseweb="tab"]{
      background-color:rgba(255,255,255,0.05);
      border:1px solid rgba(255,255,255,0.1);
      border-radius:12px;padding:10px 22px;
      color:#e5e7eb;font-weight:600;
  }
  .stTabs [data-baseweb="tab"]:hover{background-color:rgba(22,163,74,0.2);}
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
        data=json.dumps({"message": f"correction: {wrong} → {correct}",
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

    body = (f"**Wrong prediction:** {wrong}\n\n"
            f"**Correct label:** {correct}\n\n"
            + (f"[📷 Image]({img_url})\n\n" if img_url else "")
            + "*Automatic correction from user*")
    req2 = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues",
        data=json.dumps({"title": f"Correction: {wrong} → {correct}",
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
    if p_med > 0.65:  return "🟢 Medicinal",        p_med
    if p_med < 0.35:  return "🔴 Non-medicinal",     1 - p_med
    return               "🟡 Uncertain",             max(p_med, 1-p_med)

def label_v4(p_med):
    if p_med >= 0.5:
        return "🟢 Usage médical important", p_med
    else:
        return "🔴 Usage médical quasi inexistant", 1 - p_med

def bar(label, pct, green=True):
    cls = "green-fill" if green else "gray-fill"
    return f"""
    <div class="bar-wrap">
      <div class="bar-label">{label}</div>
      <div class="bar-track">
        <div class="bar-fill {cls}" style="width:{pct*100:.1f}%">
          {pct*100:.1f}%
        </div>
      </div>
    </div>"""

for k in ("results","feedback_sent","img_bytes"):
    if k not in st.session_state:
        st.session_state[k] = None

st.markdown("""
<div style="background:linear-gradient(135deg,#14532d,#16a34a 55%,#4ade80);
            border-radius:24px;padding:35px 20px;text-align:center;color:white;
            margin-bottom:25px;box-shadow:0 15px 30px -5px rgba(22,163,74,0.3);">
  <h1 style="margin:0 0 8px 0;font-size:2.1rem;color:white;
             text-shadow:0 2px 4px rgba(0,0,0,0.3);">
    🌿 Medicinal Plant Classifier</h1>
  <p style="margin:0 0 16px 0;font-size:1.1rem;color:white;opacity:.95;">
    Upload or take a photo of a plant — the three models will analyse it</p>
  <span style="display:inline-block;background:rgba(255,255,255,0.2);
                border:1px solid rgba(255,255,255,0.5);border-radius:999px;
                padding:8px 22px;font-size:1.05rem;color:white;">
      ✨ Created by: <b>Hamza Oulbayyad</b></span>
</div>
""", unsafe_allow_html=True)

col_results, col_input = st.columns([2, 1])

with col_input:
    st.markdown("#### 📸 Image of the plant")
    tab_up, tab_cam = st.tabs(["📁 Upload", "📷 Camera"])
    with tab_up:
        uploaded = st.file_uploader("", type=["jpg","jpeg","png","webp","jfif"],
                                    label_visibility="collapsed")
    with tab_cam:
        camera = st.camera_input("")
    img_source = uploaded or camera

    if img_source:
        image = Image.open(img_source)
        image = ImageOps.exif_transpose(image)
        st.image(image, use_container_width=True)

    if st.button("🔬 Try it now", type="primary", use_container_width=True):
        if img_source is None:
            st.warning("Please upload or take a photo first.")
        else:
            with st.spinner("Analysing..."):
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
        c1, c2, c3 = st.columns(3)

        for col, model_name, model_sub, p_med, lbl_fn in [
            (c1, "Model 1", "Medical / Non-medical", p1, label_v1v2),
            (c2, "Model 2", "Medical / Non-medical", p2, label_v1v2),
            (c3, "Model 4", "Strong medical use / almost non-existent", p4, label_v4),
        ]:
            lbl, conf = lbl_fn(p_med)
            is_green = "🟢" in lbl
            with col:
                st.markdown(f"""
                <div class="model-card">
                  <div class="model-title">{model_name}</div>
                  <div class="model-sub">{model_sub}</div>
                  {bar(lbl, conf, green=is_green)}
                </div>""", unsafe_allow_html=True)

        st.divider()
        if not st.session_state.feedback_sent:
            st.markdown("**Is Model 4's prediction correct?**")
            fc1, fc2 = st.columns(2)
            with fc1:
                if st.button("✅ Yes, correct", use_container_width=True):
                    st.session_state.feedback_sent = "correct"
                    st.rerun()
            with fc2:
                if st.button("❌ No, wrong", use_container_width=True):
                    st.session_state.feedback_sent = "wrong"
                    st.rerun()

        elif st.session_state.feedback_sent == "correct":
            st.success("Thanks for confirming! 🎉")

        elif st.session_state.feedback_sent == "wrong":
            st.markdown("**What is the correct label?**")
            fix = st.selectbox("", ["", "Medicinal", "Non-medicinal"],
                                label_visibility="collapsed")
            if fix and st.button("📨 Submit correction", use_container_width=True):
                p1, p2, p4 = st.session_state.results
                lbl4, _ = label_v4(p4)
                with st.spinner("Saving..."):
                    try:
                        save_correction_to_github(
                            lbl4, fix, st.session_state.img_bytes)
                        st.session_state.feedback_sent = "done"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        elif st.session_state.feedback_sent == "done":
            st.success("Saved! Thank you 🎉")
            st.info("This correction will be used to improve the model later.")

    else:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:center;
                    height:300px;opacity:.4;font-size:1.1rem;">
          Upload a plant photo and press <b>&nbsp;Try it now&nbsp;</b>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;font-size:.85rem;color:#9ca3af;margin-top:35px;
            border-top:1px solid rgba(255,255,255,0.1);padding-top:15px;">
  ⚠️ Educational project. Do NOT use to decide if a plant is safe to eat or use medicinally.<br>
  <b>Hamza Oulbayyad</b> — Data: PFAF database (pfaf.org)
</div>
""", unsafe_allow_html=True)
