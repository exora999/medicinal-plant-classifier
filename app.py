import json
import urllib.request
import urllib.parse
import html
import uuid
import base64
import io as _io
import time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras
import os
import gdown

st.set_page_config(
    page_title="Medicinal Plant Classifier — Hamza Oulbayyad & Rida Bargach",
    page_icon="🌿",
    layout="wide"
)

LANGUAGES = {
    "🇬🇧 English": "en",
    "🇫🇷 Français": "fr",
    "🇲🇦 العربية": "ar"
}

TEXT = {
    "en": {
        "lang_label": "Display Language",
        "hero": "Medicinal Plant Classifier",
        "subtitle": "Compare 4 deep learning architectures — Model 1, Model 2, Model 4, and Model 4 Pro",
        "developed": "✨ Developed by:",
        "input": "📸 Input Plant Image",
        "upload_tab": "📁 Upload File",
        "camera_tab": "📷 Take Photo",
        "upload": "Upload plant image",
        "capture": "Capture plant photo",
        "ready_img": "✅ Image Ready",
        "analyse": "📊  |  Analyse Plant  ➔",
        "upload_first": "Please upload or capture a plant photo first.",
        "identifying": "🌿 Identifying plant with PlantNet...",
        "fetching": "🧪 Retrieving chemical compounds...",
        "translating": "🌐 Translating botanical information...",
        "running": "Running inference across all 4 models...",
        "plant_id": "Plant Identification",
        "scientific": "Scientific Name",
        "common": "Common Name",
        "confidence": "PlantNet Confidence",
        "not_found": "PlantNet could not identify this plant. Try a clearer photo.",
        "plant_info": "Plant Information (PFAF)",
        "family": "🌱 Family",
        "rating": "⭐ Medicinal Rating (0–5)",
        "range": "🗺️ Native Range",
        "hazards": "⚠️ Known Hazards",
        "properties": "💊 Medicinal Properties",
        "summary": "📝 Summary",
        "not_db": "This plant was identified by PlantNet, but no matching entry was found in pfaf_plants_small.csv.",
        "csv_missing": "The CSV file pfaf_plants_small.csv was not found beside the application.",
        "chemicals": "Chemical Compounds (PubChem)",
        "chem_note": "Chemical compound names are kept in their standard scientific form.",
        "no_chem": "No chemical compound names were returned by PubChem.",
        "plantnet_error": "PlantNet identification was unavailable: ",
        "baseline": "Baseline Architecture",
        "intermediate": "Intermediate Architecture",
        "model4_lbl": "Model 4 (Standard)",
        "model4_pro_lbl": "Model 4 Pro (Optimized)",
        "primary": "⭐ Pro Model",
        "evaluate": "🎯 Evaluate Model 4 Pro",
        "evaluate_help": "Is <b>Model 4 Pro's</b> classification accurate for this plant?",
        "yes": "✅ Yes, Model 4 Pro is correct",
        "no": "❌ No, Model 4 Pro is wrong",
        "thanks": "Thank you for validating Model 4 Pro's prediction! 🎉",
        "select_true": "Select the true classification for this plant:",
        "correct_label": "Correct label",
        "medicinal": "Medicinal",
        "nonmed": "Non-medicinal",
        "submit": "📨 Submit Correction",
        "saving": "Saving correction to dataset...",
        "saved": "Correction saved! Thank you 🎉",
        "save_error": "Error saving feedback: ",
        "ready": "Ready to Analyse",
        "ready_help": "Upload a plant photo on the right and click <b>Analyse Plant</b>",
        "academic": "⚠️ <b>Academic & Educational Project</b> — Do NOT use this tool for medical diagnosis or consumption decisions.",
        "dataset": "Dataset: Plants For A Future (pfaf.org)",
        "model1": "Model 1",
        "model2": "Model 2",
        "model3": "Model 4",
        "model4_name": "Model 4 Pro",
        "med_result": "🟢 Medicinal",
        "nonmed_result": "🔴 Non-medicinal"
    },
    "fr": {
        "lang_label": "Langue d'affichage",
        "hero": "Classification des plantes médicinales",
        "subtitle": "Comparez 4 architectures — Modèle 1, Modèle 2, Modèle 4 et Modèle 4 Pro",
        "developed": "✨ Développé par :",
        "input": "📸 Image de la plante",
        "upload_tab": "📁 Importer un fichier",
        "camera_tab": "📷 Prendre une photo",
        "upload": "Importer une image de plante",
        "capture": "Photographier une plante",
        "ready_img": "✅ Image prête",
        "analyse": "📊  |  Analyser la plante  ➔",
        "upload_first": "Veuillez importer ou prendre une photo d’une plante.",
        "identifying": "🌿 Identification avec PlantNet...",
        "fetching": "🧪 Récupération des composés chimiques...",
        "translating": "🌐 Traduction des informations botaniques...",
        "running": "Exécution des prédictions sur les 4 modèles...",
        "plant_id": "Identification de la plante",
        "scientific": "Nom scientifique",
        "common": "Nom commun",
        "confidence": "Confiance PlantNet",
        "not_found": "PlantNet n’a pas pu identifier cette plante.",
        "plant_info": "Informations sur la plante (PFAF)",
        "family": "🌱 Famille",
        "rating": "⭐ Note médicinale (0–5)",
        "range": "🗺️ Répartition naturelle",
        "hazards": "⚠️ Dangers connus",
        "properties": "💊 Propriétés médicinales",
        "summary": "📝 Résumé",
        "not_db": "Aucune entrée correspondante trouvée dans pfaf_plants_small.csv.",
        "csv_missing": "Le fichier CSV est introuvable.",
        "chemicals": "Composés chimiques (PubChem)",
        "chem_note": "Noms scientifiques normalisés.",
        "no_chem": "Aucun composé chimique trouvé.",
        "plantnet_error": "PlantNet indisponible : ",
        "baseline": "Architecture de base",
        "intermediate": "Architecture intermédiaire",
        "model4_lbl": "Modèle 4 (Standard)",
        "model4_pro_lbl": "Modèle 4 Pro (Optimisé)",
        "primary": "⭐ Modèle Pro",
        "evaluate": "🎯 Évaluer le Modèle 4 Pro",
        "evaluate_help": "La classification du <b>Modèle 4 Pro</b> est-elle correcte ?",
        "yes": "✅ Oui, correct",
        "no": "❌ Non, incorrect",
        "thanks": "Merci !",
        "select_true": "Sélectionnez la classification réelle :",
        "correct_label": "Étiquette correcte",
        "medicinal": "Médicinale",
        "nonmed": "Non médicinale",
        "submit": "📨 Envoyer la correction",
        "saving": "Enregistrement...",
        "saved": "Corrigé avec succès !",
        "save_error": "Erreur : ",
        "ready": "Prêt à analyser",
        "ready_help": "Importez une photo puis cliquez sur <b>Analyser la plante</b>",
        "academic": "⚠️ <b>Projet académique</b>",
        "dataset": "Base de données : PFAF",
        "model1": "Modèle 1",
        "model2": "Modèle 2",
        "model3": "Modèle 4",
        "model4_name": "Modèle 4 Pro",
        "med_result": "🟢 Médicinale",
        "nonmed_result": "🔴 Non médicinale"
    },
    "ar": {
        "lang_label": "لغة العرض",
        "hero": "تصنيف النباتات الطبية",
        "subtitle": "قارن بين 4 بنيات للتعلم العميق — النموذج 1، النموذج 2، النموذج 4، والنموذج 4 المحترف (Pro)",
        "developed": "✨ تطوير:",
        "input": "📸 صورة النبتة",
        "upload_tab": "📁 رفع ملف",
        "camera_tab": "📷 التقاط صورة",
        "upload": "ارفع صورة النبتة",
        "capture": "التقط صورة للنبتة",
        "ready_img": "✅ الصورة جاهزة",
        "analyse": "📊  |  تحليل النبتة  ➔",
        "upload_first": "يرجى رفع صورة النبتة أو التقاطها أولاً.",
        "identifying": "🌿 جارٍ التعرف على النبتة باستخدام PlantNet...",
        "fetching": "🧪 جارٍ جلب المركبات الكيميائية...",
        "translating": "🌐 جارٍ ترجمة المعلومات النباتية...",
        "running": "جارٍ تشغيل النماذج الأربعة...",
        "plant_id": "التعرف على النبتة",
        "scientific": "الاسم العلمي",
        "common": "الاسم الشائع",
        "confidence": "نسبة ثقة PlantNet",
        "not_found": "لم يتمكن PlantNet من تحديد النبتة.",
        "plant_info": "معلومات النبتة (PFAF)",
        "family": "🌱 الفصيلة",
        "rating": "⭐ التقييم الطبي (0–5)",
        "range": "🗺️ الموطن الأصلي",
        "hazards": "⚠️ المخاطر المعروفة",
        "properties": "💊 الخصائص الطبية",
        "summary": "📝 ملخص",
        "not_db": "لم يُعثر على معلومات مطابقة في قاعدة البيانات.",
        "csv_missing": "ملف قاعدة البيانات غير موجود.",
        "chemicals": "المركبات الكيميائية (PubChem)",
        "chem_note": "تُترك أسماء المركبات بصيغتها العلمية.",
        "no_chem": "لم يُرجع PubChem أسماء للمركبات.",
        "plantnet_error": "خدمة PlantNet غير متاحة: ",
        "baseline": "البنية الأساسية",
        "intermediate": "البنية المتوسطة",
        "model4_lbl": "النموذج 4 (العادي)",
        "model4_pro_lbl": "النموذج 4 المحترف (Pro)",
        "primary": "⭐ نموذج Pro الرئيسي",
        "evaluate": "🎯 تقييم النموذج 4 Pro",
        "evaluate_help": "هل تصنيف <b>النموذج 4 Pro</b> صحيح لهذه النبتة؟",
        "yes": "✅ نعم، صحيح",
        "no": "❌ لا، غير صحيح",
        "thanks": "شكراً لتأكيد النتيجة! 🎉",
        "select_true": "اختر التصنيف الصحيح لهذه النبتة:",
        "correct_label": "التصنيف الصحيح",
        "medicinal": "طبية",
        "nonmed": "غير طبية",
        "submit": "📨 إرسال التصحيح",
        "saving": "جارٍ حفظ التصحيح...",
        "saved": "تم حفظ التصحيح بنجاح! 🎉",
        "save_error": "خطأ أثناء الحفظ: ",
        "ready": "جاهز للتحليل",
        "ready_help": "ارفع صورة النبتة ثم اضغط على <b>تحليل النبتة</b>",
        "academic": "⚠️ <b>مشروع أكاديمي وتعليمي</b> — لا تستخدم لأغراض التشخيص الطبي.",
        "dataset": "قاعدة البيانات: PFAF",
        "model1": "النموذج 1",
        "model2": "النموذج 2",
        "model3": "النموذج 4",
        "model4_name": "النموذج 4 Pro",
        "med_result": "🟢 طبية",
        "nonmed_result": "🔴 غير طبية"
    }
}

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Cairo:wght@500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', 'Cairo', sans-serif; }
  header[data-testid="stHeader"] { background: transparent !important; }
  .block-container { padding-top: 2.2rem !important; }
  .stApp {
      background: radial-gradient(circle at 50% -10%, #064e3b 0%, #0f172a 55%, #020617 100%);
      color: #f8fafc;
  }
  .top-lang-badge {
      display: inline-flex; align-items: center; gap: 8px;
      background: rgba(6, 78, 59, 0.45); border: 1px solid rgba(74, 222, 128, 0.3);
      border-radius: 999px; padding: 7px 16px; font-size: 0.82rem; font-weight: 700; color: #a7f3d0;
  }
  div[data-testid="stSelectbox"] > label { display: none !important; }
  div[data-testid="stSelectbox"] > div > div {
      background: linear-gradient(135deg, rgba(6, 78, 59, 0.65), rgba(15, 23, 42, 0.85)) !important;
      border: 1px solid rgba(74, 222, 128, 0.42) !important; border-radius: 14px !important; color: #f8fafc !important; font-weight: 700 !important;
  }
  .hero-box {
      background: linear-gradient(135deg, rgba(6, 78, 59, 0.85), rgba(22, 163, 74, 0.75), rgba(16, 185, 129, 0.65));
      border: 1px solid rgba(74, 222, 128, 0.35); border-radius: 26px; padding: 34px 24px; text-align: center; color: white; margin-bottom: 28px;
  }
  .hero-badge {
      display: inline-block; background: rgba(255, 255, 255, 0.15); border: 1px solid rgba(255, 255, 255, 0.4);
      border-radius: 999px; padding: 8px 22px; font-size: 0.95rem; color: #ffffff; margin-top: 10px;
  }
  .model-card {
      background: rgba(30, 41, 59, 0.55); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 22px; padding: 20px 14px; text-align: center; height: 100%;
  }
  .model-card-primary {
      background: linear-gradient(180deg, rgba(6, 78, 59, 0.45) 0%, rgba(15, 23, 42, 0.8) 100%);
      border: 1.5px solid #4ade80; border-radius: 22px; padding: 20px 14px; text-align: center; height: 100%;
      box-shadow: 0 10px 30px -10px rgba(74, 222, 128, 0.25);
  }
  .primary-tag {
      display: inline-block; background: rgba(74, 222, 128, 0.18); color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.4);
      font-size: 0.75rem; font-weight: 700; padding: 3px 12px; border-radius: 999px; margin-bottom: 10px; text-transform: uppercase;
  }
  .secondary-tag {
      display: inline-block; background: rgba(148, 163, 184, 0.15); color: #94a3b8; font-size: 0.75rem; font-weight: 600; padding: 3px 12px; border-radius: 999px; margin-bottom: 10px;
  }
  .model-title { font-size: 1.15rem; font-weight: 800; color: #f8fafc; margin-bottom: 4px; }
  .model-sub { font-size: 0.8rem; color: #94a3b8; margin-bottom: 14px; }
  .bar-wrap { margin: 8px 0 4px 0; }
  .bar-label { font-size: 0.95rem; font-weight: 700; margin-bottom: 6px; }
  .bar-track { background: rgba(15, 23, 42, 0.8); border-radius: 999px; height: 22px; width: 100%; overflow: hidden; border: 1px solid rgba(255, 255, 255, 0.08); padding: 2px; }
  .bar-fill { height: 16px; border-radius: 999px; display: flex; align-items: center; padding-right: 8px; justify-content: flex-end; color: white; font-size: 0.78rem; font-weight: 800; }
  .green-fill { background: linear-gradient(90deg, #15803d, #22c55e, #4ade80); }
  .red-fill   { background: linear-gradient(90deg, #991b1b, #ef4444, #f87171); }
  .amber-fill { background: linear-gradient(90deg, #b45309, #f59e0b, #fbbf24); }
  .feedback-box { background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(74, 222, 128, 0.25); border-radius: 18px; padding: 18px 22px; margin-top: 16px; }
  .stButton>button { border-radius: 14px !important; font-weight: 700 !important; background: rgba(30, 41, 59, 0.8) !important; border: 1px solid rgba(255, 255, 255, 0.15) !important; color: #ffffff !important; width: 100% !important; }
  .st-key-analyse_btn button {
      background: linear-gradient(90deg, #16a34a 0%, #22c55e 50%, #10b981 100%) !important;
      border: 1.5px solid #4ade80 !important; border-radius: 999px !important; color: #ffffff !important; font-size: 1.05rem !important; font-weight: 800 !important; padding: 14px 24px !important;
  }
  .preview-wrapper { position: relative; background: #ffffff; border-radius: 22px; padding: 10px; margin-top: 12px; margin-bottom: 14px; border: 2px solid rgba(74, 222, 128, 0.35); }
  .preview-wrapper img { width: 100%; border-radius: 14px; display: block; object-fit: contain; max-height: 290px; }
  .ready-pill { position: absolute; top: 18px; right: 18px; background: rgba(220, 252, 231, 0.95); color: #15803d; border-radius: 999px; padding: 5px 14px; font-size: 0.8rem; font-weight: 800; }
  .plant-section-heading { display: flex; align-items: center; gap: 11px; margin: 20px 0 12px 0; padding: 12px 16px; border-radius: 15px; background: linear-gradient(100deg, rgba(6, 78, 59, .62), rgba(30, 41, 59, .52)); border: 1px solid rgba(74, 222, 128, .28); color: #ecfdf5; font-weight: 800; }
  .plant-info-card { background: linear-gradient(135deg, rgba(6, 78, 59, .88) 0%, rgba(13, 90, 82, .84) 52%, rgba(20, 48, 62, .92) 100%); border: 1px solid rgba(74, 222, 128, .38); border-radius: 17px; padding: 16px 19px; margin: 7px 0 13px 0; color: #ffffff; }
  .plant-info-title { font-size: .78rem; color: #a7f3d0; font-weight: 800; margin-bottom: 8px; }
  .plant-info-value { font-size: .98rem; line-height: 1.75; }
  .chemical-panel { background: linear-gradient(145deg, rgba(6, 78, 59, .3), rgba(15, 23, 42, .65)); border: 1px solid rgba(74, 222, 128, .26); border-radius: 18px; padding: 15px 17px; margin: 8px 0 16px 0; }
  .chemical-pill { display: inline-block; direction: ltr; background: linear-gradient(135deg, rgba(22, 163, 74, .24), rgba(13, 148, 136, .18)); border: 1px solid rgba(74, 222, 128, .36); border-radius: 10px; padding: 7px 12px; margin: 5px 6px 5px 0; color: #bbf7d0; font-size: .84rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

top_left, top_right = st.columns([3.1, 1.1])
with top_left:
    st.markdown('<div class="top-lang-badge">🌿 AI Phytotherapy &bull; PlantNet &bull; PFAF &bull; PubChem</div>', unsafe_allow_html=True)
with top_right:
    language_label = st.selectbox("Language", list(LANGUAGES.keys()), index=0, key="site_language", label_visibility="collapsed")

LANG = LANGUAGES[language_label]

def tr(key):
    return TEXT[LANG].get(key, TEXT["en"].get(key, key))

def _split_text_chunks(text: str, max_len: int = 1300):
    chunks = []
    remaining = text.strip()
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break
        cut = max(remaining.rfind("\n", 0, max_len), remaining.rfind(". ", 0, max_len), remaining.rfind(" ", 0, max_len))
        if cut < 200: cut = max_len
        else: cut += 1
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    return chunks

def _google_translate_chunk(chunk: str, target: str) -> str:
    params = urllib.parse.urlencode({"client": "gtx", "sl": "auto", "tl": target, "dt": "t", "q": chunk})
    url = f"https://translate.googleapis.com/translate_a/single?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    parts = [seg[0] for seg in payload[0] if seg and seg[0]]
    return "".join(parts).strip()

@st.cache_data(ttl=86400, show_spinner=False)
def _translate_cached(value: str, target: str) -> str:
    chunks = _split_text_chunks(value, max_len=1300)
    return "\n".join([_google_translate_chunk(c, target) for c in chunks])

def translate_botanical_text(value, target):
    if not value or target == "en": return value
    try: return _translate_cached(str(value).strip(), target)
    except: return value

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
        data=json.dumps({"message": f"correction (Model 4 Pro): {wrong} -> {correct}", "content": b64}).encode(),
        method="PUT",
        headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json", "Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as r:
            img_url = json.loads(r.read())["content"]["html_url"]
    except:
        img_url = None

    body = f"**Model Evaluated:** Model 4 Pro\n\n**Prediction:** {wrong}\n\n**User Correct Label:** {correct}\n\n" + (f"[📷 Plant Image]({img_url})\n\n" if img_url else "")
    req2 = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues",
        data=json.dumps({"title": f"Model 4 Pro Correction: {wrong} -> {correct}", "body": body, "labels": ["correction"]}).encode(),
        method="POST",
        headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json", "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req2) as r:
        return json.loads(r.read())["html_url"]

@st.cache_data(show_spinner=False)
def load_plant_database():
    for csv_path in [Path(__file__).resolve().parent / "pfaf_plants_small.csv", Path.cwd() / "pfaf_plants_small.csv"]:
        if csv_path.is_file():
            try: return pd.read_csv(csv_path)
            except: return pd.DataFrame()
    return pd.DataFrame()

def identify_plant(img_bytes: bytes, target_language: str = "en"):
    api_key = st.secrets.get("PLANTNET_API_KEY", "")
    if not api_key: raise RuntimeError("PLANTNET_API_KEY is missing.")
    boundary = "----PlantNetBoundary" + uuid.uuid4().hex
    query = urllib.parse.urlencode({"api-key": api_key, "include-related-images": "false", "lang": target_language if target_language in ("en", "fr", "ar") else "en"})
    url = f"https://my-api.plantnet.org/v2/identify/all?{query}"
    body = f"--{boundary}\r\nContent-Disposition: form-data; name=\"organs\"\r\n\r\nleaf\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"images\"; filename=\"plant.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    results = data.get("results", [])
    if not results: return None, None, 0.0
    top = results[0]
    species = top.get("species", {})
    sci = species.get("scientificNameWithoutAuthor") or species.get("scientificName") or "Unknown"
    comms = species.get("commonNames") or []
    return sci, (comms[0] if comms else ""), round(float(top.get("score", 0)) * 100, 1)

def find_plant_in_database(scientific_name: str):
    if not scientific_name or df_plants.empty: return None
    def norm(v): return "".join(ch.lower() for ch in str(v) if ch.isalnum())
    name_column = next((col for col in df_plants.columns if norm(col) in {"scientificname", "botanicalname", "latinname", "species", "scientific"}), None)
    if not name_column: return None
    names = df_plants[name_column].fillna("").astype(str).str.strip()
    target = norm(scientific_name)
    exact = df_plants[names.map(norm) == target]
    if not exact.empty: return exact.iloc[0].to_dict()
    target_binomial = " ".join(scientific_name.casefold().split()[:2])
    for idx, candidate in names.items():
        if " ".join(str(candidate).casefold().split()[:2]) == target_binomial:
            return df_plants.loc[idx].to_dict()
    return None

def get_chemical_compounds(scientific_name: str):
    if not scientific_name: return []
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(scientific_name)}/JSON?MaxRecords=5"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        names = []
        for compound in data.get("PC_Compounds", [])[:5]:
            for prop in compound.get("props", []):
                if prop.get("urn", {}).get("label") == "IUPAC Name":
                    val = prop.get("value", {}).get("sval", "")
                    if val: names.append(val)
                    break
        return names[:5]
    except: return []

def render_plant_information(plant_data):
    text_dir = "rtl" if LANG == "ar" else "ltr"
    st.markdown(f'<div dir="{text_dir}" class="plant-section-heading"><span class="section-mark">🌿</span>{html.escape(tr("plant_id"))}</div>', unsafe_allow_html=True)
    row = plant_data.get("database_row")
    if plant_data.get("scientific_name"):
        cols = st.columns(2, gap="medium")
        with cols[0]:
            st.markdown(f'<div dir="{text_dir}" class="plant-info-card"><div class="plant-info-title">{tr("scientific")}</div><div class="plant-info-value"><i dir="ltr">{html.escape(str(plant_data["scientific_name"]))}</i></div></div>', unsafe_allow_html=True)
        with cols[1]:
            comm = plant_data.get("common_name") or (str(row.get("Common Name") or "").strip() if row else "")
            st.markdown(f'<div dir="{text_dir}" class="plant-info-card"><div class="plant-info-title">{tr("common")}</div><div class="plant-info-value">{html.escape(str(translate_botanical_text(comm, LANG)) if comm else "—")}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div dir="{text_dir}" class="plant-info-card"><div class="plant-info-title">{tr("confidence")}</div><div class="plant-info-value"><span dir="ltr">{float(plant_data.get("confidence", 0)):.1f}%</span></div></div>', unsafe_allow_html=True)
    else:
        st.warning(tr("not_found"))

    if row:
        st.markdown(f'<div dir="{text_dir}" class="plant-section-heading"><span class="section-mark">📋</span>{html.escape(tr("plant_info"))}</div>', unsafe_allow_html=True)
        label_map = {"Family": "family", "Medicinal Rating": "rating", "Native Range": "range", "Known Hazards": "hazards", "Medicinal Properties": "properties", "Summary": "summary"}
        excluded = {"Scientific Name", "Common Name", "Plant Name", "ID", "id", "Slug", "URL", "Source", "Image"}
        cols = st.columns(2, gap="medium")
        idx = 0
        for k, v in row.items():
            if k in excluded or pd.isna(v) or not str(v).strip() or str(v).lower() in ("nan", "none"): continue
            val_str = f"{float(v):.0f} / 5" if k == "Medicinal Rating" and isinstance(v, (int, float, np.number)) else translate_botanical_text(str(v), LANG)
            lbl = tr(label_map[k]) if k in label_map else translate_botanical_text(str(k), LANG)
            with cols[idx % 2]:
                st.markdown(f'<div dir="{text_dir}" class="plant-info-card"><div class="plant-info-title">{html.escape(str(lbl))}</div><div class="plant-info-value">{html.escape(str(val_str)[:2200])}</div></div>', unsafe_allow_html=True)
            idx += 1

    chem = plant_data.get("chemicals", [])
    if chem:
        st.markdown(f'<div dir="{text_dir}" class="plant-section-heading"><span class="section-mark">🧪</span>{html.escape(tr("chemicals"))}</div>', unsafe_allow_html=True)
        st.markdown(f'<div dir="{text_dir}" class="chemical-panel"><div style="color:#a7f3d0;font-size:.86rem;margin-bottom:8px;">{html.escape(tr("chem_note"))}</div>' + "".join([f'<span dir="ltr" class="chemical-pill">{html.escape(str(c))}</span>' for c in chem]) + '</div>', unsafe_allow_html=True)

@st.cache_resource
def load_models():
    m1 = keras.models.load_model("plant_model_v1.keras", compile=False)
    m2 = keras.models.load_model("plant_model_v2.keras", compile=False)
    
    # تحميل النموذج الرابع العادي
    m4_path = "plant_model_v4.keras"
    if not os.path.exists(m4_path) or os.path.getsize(m4_path) < 1_000_000:
        try: gdown.download(f"https://drive.google.com/uc?id=1KosNp7HeNJVMYW0flYBGmVUJP7LjCRjg", m4_path, quiet=True)
        except: pass
    m4 = keras.models.load_model(m4_path, compile=False) if os.path.exists(m4_path) else None

    # تحميل نموذج Pro الجديد
    pro_path = "plant_model_v4_pro.keras"
    if not os.path.exists(pro_path) or os.path.getsize(pro_path) < 1_000_000:
        try: gdown.download(f"https://drive.google.com/uc?id=1KosNp7HeNJVMYW0flYBGmVUJP7LjCRjg", pro_path, quiet=True)
        except: pass
    m4_pro = keras.models.load_model(pro_path, compile=False) if os.path.exists(pro_path) else None

    return m1, m2, m4, m4_pro

m1, m2, m4, m4_pro = load_models()
df_plants = load_plant_database()

def predict(model, img):
    if model is None: return 0.5
    x = np.expand_dims(np.array(img.convert("RGB").resize((224, 224)), dtype="float32"), axis=0)
    p_non = float(model.predict(x, verbose=0)[0][0])
    return 1 - p_non

def label_v1v2(p):
    if p > 0.65: return tr("med_result"), p, "green-fill", "#4ade80"
    if p < 0.35: return tr("nonmed_result"), 1 - p, "red-fill", "#f87171"
    return tr("uncertain") if "uncertain" in TEXT[LANG] else "🟡 Uncertain", max(p, 1 - p), "amber-fill", "#fbbf24"

def label_v4(p):
    if p >= 0.5: return "🟢 " + tr("medicinal"), p, "green-fill", "#4ade80"
    return "🔴 " + tr("nonmed"), 1 - p, "red-fill", "#f87171"

for k in ("results", "feedback_sent", "img_bytes", "plant_data"):
    if k not in st.session_state: st.session_state[k] = None

page_dir = "rtl" if LANG == "ar" else "ltr"

st.markdown(f"""
<div class="hero-box" dir="{page_dir}">
  <h1 style="margin:0 0 10px 0;font-size:2.3rem;font-weight:800;color:white;">🌿 {tr('hero')}</h1>
  <p style="margin:0 0 14px 0;font-size:1.1rem;color:#ecfdf5;">{tr('subtitle')}</p>
  <span class="hero-badge">{tr('developed')} <b>Hamza Oulbayyad</b> &bull; <b>Rida Bargach</b></span>
</div>
""", unsafe_allow_html=True)

col_results, col_input = st.columns([2.4, 1], gap="large")

with col_input:
    st.markdown(f'<div dir="{page_dir}"><h4>{tr("input")}</h4></div>', unsafe_allow_html=True)
    tab_up, tab_cam = st.tabs([tr("upload_tab"), tr("camera_tab")])
    with tab_up: uploaded = st.file_uploader(tr("upload"), type=["jpg", "jpeg", "png", "webp", "jfif"], label_visibility="collapsed")
    with tab_cam: camera = st.camera_input(tr("capture"), label_visibility="collapsed")
    img_source = uploaded or camera

    if img_source:
        image = Image.open(img_source)
        image = ImageOps.exif_transpose(image)
        buf = _io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        st.markdown(f'<div class="preview-wrapper"><div class="ready-pill">{tr("ready_img")}</div><img src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}" /></div>', unsafe_allow_html=True)

    if st.button(tr("analyse"), key="analyse_btn"):
        if not img_source: st.warning(tr("upload_first"))
        else:
            image = ImageOps.exif_transpose(Image.open(img_source))
            buf = _io.BytesIO()
            image.convert("RGB").save(buf, format="JPEG", quality=90)
            st.session_state.img_bytes = buf.getvalue()
            
            p_data = {"scientific_name": None, "common_name": None, "confidence": 0.0, "database_row": None, "chemicals": []}
            try:
                sci, com, conf = identify_plant(st.session_state.img_bytes, LANG)
                p_data.update({"scientific_name": sci, "common_name": com, "confidence": conf})
                if sci:
                    p_data["database_row"] = find_plant_in_database(sci)
                    p_data["chemicals"] = get_chemical_compounds(sci)
            except: pass
            st.session_state.plant_data = p_data
            st.session_state.results = (predict(m1, image), predict(m2, image), predict(m4, image), predict(m4_pro, image))
            st.session_state.feedback_sent = False

with col_results:
    if st.session_state.results:
        if st.session_state.plant_data: render_plant_information(st.session_state.plant_data)
        
        p1, p2, p4, p4_pro = st.session_state.results
        c1, c2, c3, c4 = st.columns(4, gap="small")
        
        models_info = [
            (c1, tr("model1"), tr("baseline"), tr("secondary-tag"), p1, label_v1v2, False),
            (c2, tr("model2"), tr("intermediate"), tr("secondary-tag"), p2, label_v1v2, False),
            (c3, tr("model3"), tr("model4_lbl"), tr("secondary-tag"), p4, label_v4, False),
            (c4, tr("model4_name"), tr("model4_pro_lbl"), tr("primary-tag"), p4_pro, label_v4, True),
        ]
        
        for col, m_name, m_sub, tag_cls, p_val, lbl_fn, is_primary in models_info:
            lbl, conf, fill_cls, txt_col = lbl_fn(p_val)
            card_cls = "model-card-primary" if is_primary else "model-card"
            with col:
                st.markdown(f"""
                <div class="{card_cls}" dir="{page_dir}">
                  <span class="{tag_cls}">{m_sub}</span>
                  <div class="model-title">{m_name}</div>
                  <div class="bar-wrap">
                    <div class="bar-label" style="color:{txt_col}; font-size:0.85rem;">{lbl}</div>
                    <div class="bar-track"><div class="bar-fill {fill_cls}" style="width:{conf*100:.1f}%">{conf*100:.1f}%</div></div>
                  </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
        with st.container():
            st.markdown(f'<div class="feedback-box" dir="{page_dir}"><div style="font-weight:700;color:#4ade80;margin-bottom:4px;">{tr("evaluate")}</div><div style="font-size:0.9rem;color:#cbd5e1;">{tr("evaluate_help")}</div></div>', unsafe_allow_html=True)
            if not st.session_state.feedback_sent:
                fc1, fc2 = st.columns(2)
                with fc1:
                    if st.button(tr("yes")): st.session_state.feedback_sent = "correct"; st.rerun()
                with fc2:
                    if st.button(tr("no"), key="wrong_btn"): st.session_state.feedback_sent = "wrong"; st.rerun()
            elif st.session_state.feedback_sent == "correct": st.success(tr("thanks"))
            elif st.session_state.feedback_sent == "wrong":
                fix = st.selectbox(tr("correct_label"), ["", tr("medicinal"), tr("nonmed")], label_visibility="collapsed")
                if fix and st.button(tr("submit")):
                    try:
                        save_correction_to_github("Model 4 Pro Prediction", "Medicinal" if fix == tr("medicinal") else "Non-medicinal", st.session_state.img_bytes)
                        st.session_state.feedback_sent = "done"; st.rerun()
                    except Exception as e: st.error(tr("save_error") + str(e))
            elif st.session_state.feedback_sent == "done": st.success(tr("saved"))
    else:
        st.markdown(f'<div dir="{page_dir}" style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:340px;background:rgba(30,41,59,0.3);border:1px dashed rgba(255,255,255,0.15);border-radius:22px;color:#94a3b8;text-align:center;padding:20px;"><div style="font-size:2.5rem;margin-bottom:10px;">🌱</div><div style="font-size:1.1font-weight:600;color:#e2e8f0;">{tr("ready")}</div><div style="font-size:0.95rem;margin-top:4px;">{tr("ready_help")}</div></div>', unsafe_allow_html=True)

st.markdown(f'<div dir="{page_dir}" style="text-align:center;font-size:.85rem;color:#94a3b8;margin-top:40px;border-top:1px solid rgba(255,255,255,0.08);padding-top:18px;">{tr("academic")}<br><b>Hamza Oulbayyad & Rida Bargach</b> • {tr("dataset")} • <b>Version 2.0</b></div>', unsafe_allow_html=True)
