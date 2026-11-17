"""
Medicinal Plant Classifier + Encyclopedia
PlantNet identifies the plant → CSV gives info → Models classify
Author: Hamza Oulbayyad
"""
import json, urllib.request, urllib.parse, base64, io as _io, time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medicinal Plant Classifier — Hamza Oulbayyad",
    page_icon="🌿", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp {
      background: radial-gradient(circle at 50% 0%,
          #052e16 0%, #0f172a 60%, #020617 100%);
      color: #f3f4f6;
  }
  .info-card {
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 16px; padding: 18px 16px; margin-bottom: 12px;
  }
  .info-title {
      font-size: .85rem; color: #6b7280;
      text-transform: uppercase; letter-spacing: .05em; margin-bottom: 4px;
  }
  .info-value { font-size: 1rem; color: #f3f4f6; }
  .model-card {
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 18px; padding: 20px 16px; text-align: center;
  }
  .model-title { font-size: 1.1rem; font-weight: 700; color: #4ade80; margin-bottom: 4px; }
  .model-sub   { font-size: .85rem; color: #9ca3af; margin-bottom: 14px; }
  .bar-wrap  { margin: 8px 0 12px 0; }
  .bar-label { font-size: 1rem; font-weight: 600; margin-bottom: 5px; color: #4ade80; }
  .bar-track { background: rgba(255,255,255,0.08); border-radius: 999px;
               height: 22px; width: 100%; overflow: hidden;
               border: 1px solid rgba(255,255,255,0.05); }
  .bar-fill  { height: 22px; border-radius: 999px; display: flex;
               align-items: center; padding-right: 10px;
               justify-content: flex-end; color: white;
               font-size: .85rem; font-weight: 700; min-width: 48px; }
  .green-fill { background: linear-gradient(90deg,#14532d,#16a34a,#4ade80); }
  .gray-fill  { background: linear-gradient(90deg,#374151,#4b5563,#9ca3af); }
  .stButton>button {
      border-radius: 14px !important; font-weight: 700 !important;
      background: rgba(255,255,255,0.05) !important;
      border: 1px solid rgba(255,255,255,0.15) !important;
      color: #ffffff !important; transition: all 0.3s ease !important;
  }
  .stButton>button:hover {
      background: linear-gradient(135deg,#16a34a,#14532d) !important;
      border-color: #4ade80 !important; transform: translateY(-2px) !important;
      box-shadow: 0 10px 20px -3px rgba(22,163,74,0.4) !important;
  }
  .stTabs [data-baseweb="tab"] {
      background-color: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 12px; padding: 10px 22px;
      color: #e5e7eb; font-weight: 600;
  }
  .stTabs [data-baseweb="tab"]:hover {
      background-color: rgba(22,163,74,0.2);
  }
  .chem-pill {
      display: inline-block; background: rgba(74,222,128,0.15);
      border: 1px solid rgba(74,222,128,0.3); border-radius: 999px;
      padding: 4px 12px; margin: 3px; font-size: .85rem; color: #4ade80;
  }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    m1 = keras.models.load_model("plant_model_v1.keras")
    m2 = keras.models.load_model("plant_model_v2.keras")
    m4 = keras.models.load_model("plant_model_v4.keras")
    return m1, m2, m4

@st.cache_data
def load_csv():
    return pd.read_csv("pfaf_plants_small.csv")

m1, m2, m4 = load_models()
df_plants   = load_csv()

# ── PlantNet API ──────────────────────────────────────────────────────────────
def identify_plant(img_bytes: bytes):
    api_key = st.secrets["PLANTNET_API_KEY"]
    url = (f"https://my-api.plantnet.org/v2/identify/all"
           f"?api-key={api_key}&include-related-images=false&lang=en")
    boundary = "----FormBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="organs"\r\n\r\nleaf\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="images"; filename="plant.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req) as res:
        data = json.loads(res.read())
    results = data.get("results", [])
    if not results:
        return None, None, 0.0
    top = results[0]
    sci = top["species"]["scientificNameWithoutAuthor"]
    common = top["species"].get("commonNames", [])
    common_name = common[0] if common else ""
    score = round(top["score"] * 100, 1)
    return sci, common_name, score

# ── Search CSV ────────────────────────────────────────────────────────────────
def search_plant(sci_name: str):
    if "Scientific Name" not in df_plants.columns:
        return None
    mask = df_plants["Scientific Name"].str.lower().str.contains(
        sci_name.lower().split()[0], na=False)
    rows = df_plants[mask]
    return rows.iloc[0] if len(rows) else None

# ── PubChem API ───────────────────────────────────────────────────────────────
def get_chemicals(sci_name: str):
    try:
        name_enc = urllib.parse.quote(sci_name)
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
               f"name/{name_enc}/JSON?MaxRecords=5")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as res:
            data = json.loads(res.read())
        compounds = data.get("PC_Compounds", [])
        names = []
        for c in compounds[:5]:
            for prop in c.get("props", []):
                if prop.get("urn", {}).get("label") == "IUPAC Name":
                    names.append(prop["value"].get("sval", ""))
                    break
        return [n for n in names if n][:5]
    except Exception:
        return []

# ── Model helpers ─────────────────────────────────────────────────────────────
def predict_proba(model, img):
    x = np.expand_dims(
        np.array(img.convert("RGB").resize((224,224)), dtype="float32"), axis=0)
    return 1 - float(model.predict(x, verbose=0)[0][0])

def label_v1v2(p):
    if p > .65: return "🟢 Medicinal",             p
    if p < .35: return "🔴 Non-medicinal",          1-p
    return              "🟡 Uncertain",              max(p, 1-p)

def label_v4(p):
    if p > .65: return "🟢 Strong medical use",     p
    if p < .35: return "🔴 Almost no medical use",  1-p
    return              "🟡 Uncertain",              max(p, 1-p)

def bar_html(label, conf, green=True):
    cls = "green-fill" if green else "gray-fill"
    return f"""
    <div class="bar-wrap">
      <div class="bar-label">{label}</div>
      <div class="bar-track">
        <div class="bar-fill {cls}" style="width:{conf*100:.1f}%">
          {conf*100:.1f}%
        </div>
      </div>
    </div>"""

# ── GitHub correction ─────────────────────────────────────────────────────────
def save_correction(wrong: str, correct: str, img_bytes: bytes):
    token = st.secrets["GITHUB_TOKEN"]
    repo  = st.secrets["GITHUB_REPO"]
    img   = Image.open(_io.BytesIO(img_bytes)).convert("RGB").resize((224,224))
    buf   = _io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64   = base64.b64encode(buf.getvalue()).decode()
    fname = f"training_corrections/{int(time.time())}.jpg"
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/contents/{fname}",
        data=json.dumps({"message": f"correction: {wrong}→{correct}",
                         "content": b64}).encode(), method="PUT",
        headers={"Authorization": f"token {token}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            img_url = json.loads(r.read())["content"]["html_url"]
    except Exception:
        img_url = None
    body = (f"**Wrong:** {wrong}\n\n**Correct:** {correct}\n\n"
            + (f"[📷 Image]({img_url})\n\n" if img_url else "")
            + "*Auto correction from user*")
    req2 = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues",
        data=json.dumps({"title": f"Correction: {wrong}→{correct}",
                         "body": body, "labels":["correction"]}).encode(),
        method="POST",
        headers={"Authorization": f"token {token}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req2) as r:
        return json.loads(r.read())["html_url"]

# ── Session state ─────────────────────────────────────────────────────────────
for k in ("analysis","feedback_sent","img_bytes"):
    if k not in st.session_state:
        st.session_state[k] = None

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#14532d,#16a34a 55%,#4ade80);
            border-radius:24px;padding:35px 20px;text-align:center;color:white;
            margin-bottom:25px;box-shadow:0 15px 30px -5px rgba(22,163,74,.3);">
  <h1 style="margin:0 0 8px 0;font-size:2.1rem;color:white;
             text-shadow:0 2px 4px rgba(0,0,0,.3);">
    🌿 Medicinal Plant Classifier</h1>
  <p style="margin:0 0 16px 0;font-size:1.1rem;color:white;opacity:.95;">
    Upload a plant photo — get identification, medical info & chemical compounds</p>
  <span style="display:inline-block;background:rgba(255,255,255,.2);
               border:1px solid rgba(255,255,255,.5);border-radius:999px;
               padding:8px 22px;font-size:1.05rem;color:white;">
     ✨ Created by: <b>Hamza Oulbayyad</b></span>
</div>
""", unsafe_allow_html=True)

# ── Input column ──────────────────────────────────────────────────────────────
col_in, col_out = st.columns([1, 2])

with col_in:
    st.markdown("#### 📸 Plant Image")
    tab_up, tab_cam = st.tabs(["📁 Upload", "📷 Camera"])
    with tab_up:
        uploaded = st.file_uploader("", type=["jpg","jpeg","png","webp"],
                                    label_visibility="collapsed")
    with tab_cam:
        camera = st.camera_input("")
    img_source = uploaded or camera

    if img_source:
        image = Image.open(img_source)
        st.image(image, use_container_width=True)

    if st.button("🔬 Analyse Plant", type="primary", use_container_width=True):
        if not img_source:
            st.warning("Please upload or take a photo first.")
        else:
            image = Image.open(img_source)
            buf = _io.BytesIO()
            image.convert("RGB").save(buf, format="JPEG")
            img_bytes = buf.getvalue()
            st.session_state.img_bytes = img_bytes

            with st.spinner("🌿 Identifying plant with PlantNet..."):
                try:
                    sci, common, score = identify_plant(img_bytes)
                except Exception as e:
                    sci, common, score = None, None, 0
                    st.warning(f"PlantNet error: {e}")

            plant_row = search_plant(sci) if sci else None

            with st.spinner("🧪 Getting chemical compounds..."):
                chemicals = get_chemicals(sci) if sci else []

            with st.spinner("🤖 Running models..."):
                p1 = predict_proba(m1, image)
                p2 = predict_proba(m2, image)
                p4 = predict_proba(m4, image)

            st.session_state.analysis = {
                "sci": sci, "common": common, "score": score,
                "row": plant_row, "chemicals": chemicals,
                "p1": p1, "p2": p2, "p4": p4,
            }
            st.session_state.feedback_sent = False

# ── Results ───────────────────────────────────────────────────────────────────
with col_out:
    a = st.session_state.analysis
    if a is None:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:center;
                    height:300px;opacity:.4;font-size:1.1rem;">
          Upload a photo and press <b>&nbsp;Analyse Plant&nbsp;</b>
        </div>""", unsafe_allow_html=True)
    else:
        # 1) Plant identity
        st.markdown("### 🌿 Plant Identification")
        if a["sci"]:
            id_c1, id_c2 = st.columns(2)
            with id_c1:
                st.markdown(f"""
                <div class="info-card">
                  <div class="info-title">Scientific Name</div>
                  <div class="info-value"><i>{a['sci']}</i></div>
                </div>""", unsafe_allow_html=True)
            with id_c2:
                st.markdown(f"""
                <div class="info-card">
                  <div class="info-title">Common Name</div>
                  <div class="info-value">{a['common'] or '—'}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-card">
              <div class="info-title">PlantNet Confidence</div>
              <div class="info-value">{a['score']}%</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.warning("Could not identify the plant. Try a clearer photo.")

        # 2) PFAF info
        row = a["row"]
        if row is not None:
            st.markdown("### 📋 Plant Information (PFAF)")
            info_cols = st.columns(2)
            fields = [
                ("Family",             "🌱 Family"),
                ("Medicinal Rating",   "⭐ Medicinal Rating (0–5)"),
                ("Native Range",       "🗺️ Native Range"),
                ("Known Hazards",      "⚠️ Known Hazards"),
            ]
            for i, (col_name, label) in enumerate(fields):
                val = str(row.get(col_name, "—"))
                if val in ("nan",""):
                    val = "—"
                with info_cols[i % 2]:
                    st.markdown(f"""
                    <div class="info-card">
                      <div class="info-title">{label}</div>
                      <div class="info-value">{val[:300]}</div>
                    </div>""", unsafe_allow_html=True)

            med = str(row.get("Medicinal Properties","")).strip()
            if med and med != "nan":
                st.markdown(f"""
                <div class="info-card">
                  <div class="info-title">💊 Medicinal Properties</div>
                  <div class="info-value">{med[:500]}</div>
                </div>""", unsafe_allow_html=True)

            summ = str(row.get("Summary","")).strip()
            if summ and summ != "nan":
                st.markdown(f"""
                <div class="info-card">
                  <div class="info-title">📝 Summary</div>
                  <div class="info-value">{summ[:500]}</div>
                </div>""", unsafe_allow_html=True)

        # 3) Chemical compounds
        if a["chemicals"]:
            st.markdown("### 🧪 Chemical Compounds (PubChem)")
            pills = " ".join(
                f'<span class="chem-pill">{c}</span>'
                for c in a["chemicals"])
            st.markdown(pills, unsafe_allow_html=True)

        # 4) Model results
        st.markdown("### 🤖 Model Predictions")
        mc1, mc2, mc3 = st.columns(3)
        for col, name, sub, p, lbl_fn in [
            (mc1, "Model 1", "Medical / Non-medical", a["p1"], label_v1v2),
            (mc2, "Model 2", "Medical / Non-medical", a["p2"], label_v1v2),
            (mc3, "Model 4", "Strong / Almost none",  a["p4"], label_v4),
        ]:
            lbl, conf = lbl_fn(p)
            with col:
                st.markdown(f"""
                <div class="model-card">
                  <div class="model-title">{name}</div>
                  <div class="model-sub">{sub}</div>
                  {bar_html(lbl, conf, "🟢" in lbl)}
                </div>""", unsafe_allow_html=True)

        # 5) Feedback
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
            fix = st.selectbox("", ["","Medicinal","Non-medicinal"],
                               label_visibility="collapsed")
            if fix and st.button("📨 Submit correction", use_container_width=True):
                lbl4, _ = label_v4(a["p4"])
                with st.spinner("Saving..."):
                    try:
                        save_correction(lbl4, fix,
                                        st.session_state.img_bytes)
                        st.session_state.feedback_sent = "done"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        elif st.session_state.feedback_sent == "done":
            st.success("Saved! Thank you 🎉")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;font-size:.85rem;color:#9ca3af;margin-top:35px;
            border-top:1px solid rgba(255,255,255,.1);padding-top:15px;">
  ⚠️ Educational project. Do NOT use to decide if a plant is safe to eat or use medicinally.<br>
  Plant ID: <b>PlantNet API</b> — Medical data: <b>PFAF</b> —
  Chemical data: <b>PubChem</b> — Models: <b>MobileNetV2</b><br>
  <b>Hamza Oulbayyad</b>
</div>
""", unsafe_allow_html=True)
