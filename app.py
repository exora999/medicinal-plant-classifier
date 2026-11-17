import json
import urllib.request
import urllib.parse
import html
import uuid
import base64
import io as _io
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras

st.set_page_config(
    page_title="Medicinal Plant Classifier — Hamza Oulbayyad & Rida Bargach",
    page_icon="🌿",
    layout="wide"
)


# -------------------- Language selection and translations --------------------
LANGUAGES = {"English": "en", "Français": "fr", "العربية": "ar"}
TEXT = {
"en": {
"hero":"Medicinal Plant Classifier", "subtitle":"Compare three deep learning architectures — powered by <b>Model 4</b>", "developed":"✨ Developed by:",
"input":"📸 Input Plant Image", "upload_tab":"📁 Upload File", "camera_tab":"📷 Take Photo", "upload":"Upload plant image", "capture":"Capture plant photo", "ready_img":"✅ Image Ready", "analyse":"📊  |  Analyse Plant  ➔", "upload_first":"Please upload or capture a plant photo first.",
"identifying":"🌿 Identifying plant with PlantNet...", "fetching":"🧪 Retrieving chemical compounds...", "running":"Running inference across all 3 models...", "plant_id":"🌿 Plant Identification", "scientific":"Scientific Name", "common":"Common Name", "confidence":"PlantNet Confidence", "not_found":"PlantNet could not identify this plant. Try a clearer photo.", "plant_info":"📋 Plant Information (PFAF)", "family":"🌱 Family", "rating":"⭐ Medicinal Rating (0–5)", "range":"🗺️ Native Range", "hazards":"⚠️ Known Hazards", "properties":"💊 Medicinal Properties", "summary":"📝 Summary", "not_db":"This plant was identified by PlantNet, but no matching entry was found in pfaf_plants_small.csv.", "chemicals":"🧪 Chemical Compounds (PubChem)", "no_chem":"No chemical compound names were returned by PubChem.", "plantnet_error":"PlantNet identification was unavailable: ",
"baseline":"Baseline Architecture", "intermediate":"Intermediate Architecture", "strongweak":"Strong / Weak Medical Use", "primary":"⭐ Primary Model", "evaluate":"🎯 Evaluate Primary Model (Model 4)", "evaluate_help":"Ignore Models 1 & 2 above — is <b>Model 4's</b> classification accurate for this plant?", "yes":"✅ Yes, Model 4 is correct", "no":"❌ No, Model 4 is wrong", "thanks":"Thank you for validating Model 4's prediction! 🎉", "select_true":"Select the true classification for this plant:", "correct_label":"Correct label", "medicinal":"Medicinal", "nonmed":"Non-medicinal", "submit":"📨 Submit Correction for Model 4", "saving":"Saving correction to dataset...", "saved":"Correction saved! Thank you for helping improve Model 4 🎉", "save_error":"Error saving feedback: ", "ready":"Ready to Analyse", "ready_help":"Upload a plant photo on the right and click <b>Analyse Plant</b>", "academic":"⚠️ <b>Academic & Educational Project</b> — Do NOT use this tool for medical diagnosis or consumption decisions.", "dataset":"Dataset: Plants For A Future (pfaf.org)", "model1":"Model 1", "model2":"Model 2", "model4":"Model 4", "basev1":"Baseline v1", "midv2":"Intermediate v2", "uncertain":"🟡 Uncertain", "important":"🟢 Significant medical use", "almost":"🔴 Almost no medical use", "med_result":"🟢 Medicinal", "nonmed_result":"🔴 Non-medicinal"},
"fr": {
"hero":"Classification des plantes médicinales", "subtitle":"Comparez trois architectures de deep learning — propulsées par le <b>Modèle 4</b>", "developed":"✨ Développé par :", "input":"📸 Image de la plante", "upload_tab":"📁 Importer un fichier", "camera_tab":"📷 Prendre une photo", "upload":"Importer une image de plante", "capture":"Photographier une plante", "ready_img":"✅ Image prête", "analyse":"📊  |  Analyser la plante  ➔", "upload_first":"Veuillez importer ou prendre une photo d’une plante.", "identifying":"🌿 Identification avec PlantNet...", "fetching":"🧪 Récupération des composés chimiques...", "running":"Exécution des prédictions sur les 3 modèles...", "plant_id":"🌿 Identification de la plante", "scientific":"Nom scientifique", "common":"Nom commun", "confidence":"Confiance PlantNet", "not_found":"PlantNet n’a pas pu identifier cette plante. Essayez une photo plus nette.", "plant_info":"📋 Informations sur la plante (PFAF)", "family":"🌱 Famille", "rating":"⭐ Note médicinale (0–5)", "range":"🗺️ Répartition naturelle", "hazards":"⚠️ Dangers connus", "properties":"💊 Propriétés médicinales", "summary":"📝 Résumé", "not_db":"La plante a été identifiée par PlantNet, mais aucune entrée correspondante n’a été trouvée dans le fichier PFAF.", "chemicals":"🧪 Composés chimiques (PubChem)", "no_chem":"Aucun nom de composé chimique n’a été fourni par PubChem.", "plantnet_error":"L’identification PlantNet est indisponible : ", "baseline":"Architecture de base", "intermediate":"Architecture intermédiaire", "strongweak":"Usage médicinal fort / faible", "primary":"⭐ Modèle principal", "evaluate":"🎯 Évaluer le modèle principal (Modèle 4)", "evaluate_help":"Ignorez les modèles 1 et 2 — la classification du <b>Modèle 4</b> est-elle correcte ?", "yes":"✅ Oui, le Modèle 4 est correct", "no":"❌ Non, le Modèle 4 est incorrect", "thanks":"Merci d’avoir validé la prédiction du Modèle 4 ! 🎉", "select_true":"Sélectionnez la classification réelle de cette plante :", "correct_label":"Étiquette correcte", "medicinal":"Médicinale", "nonmed":"Non médicinale", "submit":"📨 Envoyer la correction du Modèle 4", "saving":"Enregistrement de la correction...", "saved":"Correction enregistrée ! Merci de contribuer à améliorer le Modèle 4 🎉", "save_error":"Erreur lors de l’enregistrement : ", "ready":"Prêt à analyser", "ready_help":"Importez une photo de plante à droite puis cliquez sur <b>Analyser la plante</b>", "academic":"⚠️ <b>Projet académique et éducatif</b> — N’utilisez PAS cet outil pour un diagnostic médical ou une décision de consommation.", "dataset":"Base de données : Plants For A Future (pfaf.org)", "model1":"Modèle 1", "model2":"Modèle 2", "model4":"Modèle 4", "basev1":"Base v1", "midv2":"Intermédiaire v2", "uncertain":"🟡 Incertain", "important":"🟢 Usage médicinal important", "almost":"🔴 Usage médicinal quasi inexistant", "med_result":"🟢 Médicinale", "nonmed_result":"🔴 Non médicinale"},
"ar": {
"hero":"تصنيف النباتات الطبية", "subtitle":"قارن بين ثلاث بنيات للتعلم العميق — بالاعتماد على <b>النموذج 4</b>", "developed":"✨ تطوير:", "input":"📸 صورة النبتة", "upload_tab":"📁 رفع ملف", "camera_tab":"📷 التقاط صورة", "upload":"ارفع صورة النبتة", "capture":"التقط صورة للنبتة", "ready_img":"✅ الصورة جاهزة", "analyse":"📊  |  تحليل النبتة  ➔", "upload_first":"يرجى رفع صورة النبتة أو التقاطها أولاً.", "identifying":"🌿 جارٍ التعرف على النبتة باستخدام PlantNet...", "fetching":"🧪 جارٍ جلب المركبات الكيميائية...", "running":"جارٍ تشغيل النماذج الثلاثة...", "plant_id":"🌿 التعرف على النبتة", "scientific":"الاسم العلمي", "common":"الاسم الشائع", "confidence":"نسبة ثقة PlantNet", "not_found":"لم يتمكن PlantNet من تحديد النبتة. جرّب صورة أوضح.", "plant_info":"📋 معلومات النبتة (PFAF)", "family":"🌱 الفصيلة", "rating":"⭐ التقييم الطبي (0–5)", "range":"🗺️ الموطن الأصلي", "hazards":"⚠️ المخاطر المعروفة", "properties":"💊 الخصائص الطبية", "summary":"📝 ملخص", "not_db":"تم التعرف على النبتة بواسطة PlantNet، لكن لم يُعثر على معلومات مطابقة في قاعدة البيانات.", "chemicals":"🧪 المركبات الكيميائية (PubChem)", "no_chem":"لم يُرجع PubChem أسماء للمركبات الكيميائية.", "plantnet_error":"خدمة PlantNet غير متاحة حالياً: ", "baseline":"البنية الأساسية", "intermediate":"البنية المتوسطة", "strongweak":"الاستخدام الطبي القوي / الضعيف", "primary":"⭐ النموذج الرئيسي", "evaluate":"🎯 تقييم النموذج الرئيسي (النموذج 4)", "evaluate_help":"تجاهل النموذجين 1 و2 — هل تصنيف <b>النموذج 4</b> صحيح لهذه النبتة؟", "yes":"✅ نعم، تصنيف النموذج 4 صحيح", "no":"❌ لا، تصنيف النموذج 4 غير صحيح", "thanks":"شكراً لتأكيد نتيجة النموذج 4! 🎉", "select_true":"اختر التصنيف الصحيح لهذه النبتة:", "correct_label":"التصنيف الصحيح", "medicinal":"طبية", "nonmed":"غير طبية", "submit":"📨 إرسال تصحيح النموذج 4", "saving":"جارٍ حفظ التصحيح...", "saved":"تم حفظ التصحيح! شكراً لمساهمتك في تحسين النموذج 4 🎉", "save_error":"خطأ أثناء حفظ الملاحظة: ", "ready":"جاهز للتحليل", "ready_help":"ارفع صورة النبتة من الجهة اليمنى ثم اضغط على <b>تحليل النبتة</b>", "academic":"⚠️ <b>مشروع أكاديمي وتعليمي</b> — لا تستخدم هذه الأداة للتشخيص الطبي أو لاتخاذ قرارات الاستهلاك.", "dataset":"قاعدة البيانات: Plants For A Future (pfaf.org)", "model1":"النموذج 1", "model2":"النموذج 2", "model4":"النموذج 4", "basev1":"الأساسي v1", "midv2":"المتوسط v2", "uncertain":"🟡 غير مؤكد", "important":"🟢 استخدام طبي مهم", "almost":"🔴 استخدام طبي شبه منعدم", "med_result":"🟢 طبية", "nonmed_result":"🔴 غير طبية"}
}

language_label = st.selectbox("🌐 Language / Langue / اللغة", list(LANGUAGES), index=0, key="site_language")
LANG = LANGUAGES[language_label]
def tr(key):
    return TEXT[LANG].get(key, TEXT["en"].get(key, key))

@st.cache_data(show_spinner=False)
def translate_botanical_text(value, target):
    """Translate descriptive database text; keep Latin scientific and chemical names unchanged."""
    if value is None or not str(value).strip() or target == "en":
        return value
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target=target).translate(str(value)[:4500]) or value
    except Exception:
        return value

if LANG == "ar":
    st.markdown("<style>.stApp, .stApp * { direction: rtl; } .stButton button { direction: rtl; }</style>", unsafe_allow_html=True)

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
      width: 100% !important;
  }
  .stButton>button:hover {
      background: linear-gradient(135deg, #16a34a, #15803d) !important;
      border-color: #4ade80 !important;
      transform: translateY(-2px) !important;
      box-shadow: 0 10px 20px -5px rgba(22, 163, 74, 0.45) !important;
  }
  .st-key-wrong_btn_trigger button:hover {
      background: linear-gradient(135deg, #dc2626, #991b1b) !important;
      border-color: #f87171 !important;
      transform: translateY(-2px) !important;
      box-shadow: 0 10px 20px -5px rgba(220, 38, 38, 0.55) !important;
  }
  .st-key-analyse_btn button {
      background: linear-gradient(90deg, #16a34a 0%, #22c55e 50%, #10b981 100%) !important;
      border: 1.5px solid #4ade80 !important;
      border-radius: 999px !important;
      color: #ffffff !important;
      font-size: 1.05rem !important;
      font-weight: 800 !important;
      padding: 14px 24px !important;
      box-shadow: 0 8px 25px -4px rgba(34, 197, 94, 0.55) !important;
      letter-spacing: 0.3px !important;
  }
  .st-key-analyse_btn button:hover {
      background: linear-gradient(90deg, #15803d 0%, #16a34a 50%, #059669 100%) !important;
      box-shadow: 0 12px 32px -2px rgba(74, 222, 128, 0.75) !important;
      transform: translateY(-2px) !important;
  }
  [data-testid="stFileUploader"] section {
      background: rgba(15, 23, 42, 0.65) !important;
      border: 1.5px dashed rgba(148, 163, 184, 0.4) !important;
      border-radius: 20px !important;
      padding: 16px !important;
      transition: border-color 0.25s ease, background 0.25s ease !important;
  }
  [data-testid="stFileUploader"] section:hover {
      border-color: #4ade80 !important;
      background: rgba(6, 78, 59, 0.25) !important;
  }
  [data-testid="stFileUploader"] section button {
      background: rgba(30, 41, 59, 0.9) !important;
      border: 1px solid rgba(74, 222, 128, 0.4) !important;
      border-radius: 12px !important;
      color: #4ade80 !important;
      font-weight: 700 !important;
  }
  .preview-wrapper {
      position: relative;
      background: #ffffff;
      border-radius: 22px;
      padding: 10px;
      margin-top: 12px;
      margin-bottom: 14px;
      box-shadow: 0 14px 32px -8px rgba(0, 0, 0, 0.6);
      border: 2px solid rgba(74, 222, 128, 0.35);
      overflow: hidden;
  }
  .preview-wrapper img {
      width: 100%;
      border-radius: 14px;
      display: block;
      object-fit: contain;
      max-height: 290px;
  }
  .ready-pill {
      position: absolute;
      top: 18px;
      right: 18px;
      background: rgba(220, 252, 231, 0.95);
      color: #15803d;
      border: 1px solid #86efac;
      border-radius: 999px;
      padding: 5px 14px;
      font-size: 0.8rem;
      font-weight: 800;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      display: flex;
      align-items: center;
      gap: 6px;
  }
  .stTabs [data-baseweb="tab-list"] {
      gap: 10px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      padding-bottom: 6px;
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
      background: linear-gradient(135deg, rgba(22, 163, 74, 0.35), rgba(6, 78, 59, 0.6)) !important;
      border-color: #4ade80 !important;
      color: #ffffff !important;
  }

  .plant-info-card {
      background: rgba(30, 41, 59, 0.55);
      border: 1px solid rgba(74, 222, 128, 0.20);
      border-radius: 16px;
      padding: 16px 18px;
      margin: 8px 0;
  }
  .plant-info-title {
      font-size: .78rem;
      color: #4ade80;
      text-transform: uppercase;
      letter-spacing: .06em;
      font-weight: 700;
      margin-bottom: 5px;
  }
  .plant-info-value {
      font-size: .96rem;
      color: #e2e8f0;
      overflow-wrap: anywhere;
  }
  .chemical-pill {
      display: inline-block;
      background: rgba(74, 222, 128, 0.13);
      border: 1px solid rgba(74, 222, 128, 0.30);
      border-radius: 999px;
      padding: 5px 12px;
      margin: 4px 5px 4px 0;
      color: #86efac;
      font-size: .82rem;
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



# ── PlantNet + PFAF + PubChem feature ─────────────────────────────────────────
@st.cache_data
def load_plant_database():
    """Load the PFAF CSV if it exists; keep the classifier usable without it."""
    try:
        return pd.read_csv("pfaf_plants_small.csv")
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def identify_plant(img_bytes: bytes):
    """Identify a plant image with the PlantNet API."""
    api_key = st.secrets.get("PLANTNET_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "PLANTNET_API_KEY is missing from Streamlit secrets."
        )

    boundary = "----PlantNetBoundary" + uuid.uuid4().hex
    query = urllib.parse.urlencode({
        "api-key": api_key,
        "include-related-images": "false",
        "lang": "en",
    })
    url = f"https://my-api.plantnet.org/v2/identify/all?{query}"

    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="organs"\r\n\r\n'
        "leaf\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="images"; filename="plant.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))

    results = data.get("results", [])
    if not results:
        return None, None, 0.0

    top = results[0]
    species = top.get("species", {})
    scientific_name = (
        species.get("scientificNameWithoutAuthor")
        or species.get("scientificName")
        or "Unknown"
    )
    common_names = species.get("commonNames") or []
    common_name = common_names[0] if common_names else ""
    confidence = round(float(top.get("score", 0)) * 100, 1)
    return scientific_name, common_name, confidence


def find_plant_in_database(scientific_name: str):
    """Find the identified species in the PFAF CSV."""
    if not scientific_name or df_plants.empty:
        return None
    if "Scientific Name" not in df_plants.columns:
        return None

    names = df_plants["Scientific Name"].fillna("").astype(str).str.strip()
    target = scientific_name.strip().casefold()

    # Prefer a full scientific-name match; fall back to the genus if necessary.
    exact = df_plants[names.str.casefold() == target]
    if not exact.empty:
        return exact.iloc[0].to_dict()

    genus = target.split()[0]
    partial = df_plants[names.str.casefold().str.startswith(genus)]
    if not partial.empty:
        return partial.iloc[0].to_dict()
    return None


def get_chemical_compounds(scientific_name: str):
    """Retrieve up to five IUPAC compound names from PubChem."""
    if not scientific_name:
        return []
    try:
        encoded_name = urllib.parse.quote(scientific_name)
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
            f"name/{encoded_name}/JSON?MaxRecords=5"
        )
        request = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        compounds = data.get("PC_Compounds", [])
        names = []
        for compound in compounds[:5]:
            for prop in compound.get("props", []):
                urn = prop.get("urn", {})
                if urn.get("label") == "IUPAC Name":
                    value = prop.get("value", {}).get("sval", "")
                    if value:
                        names.append(value)
                    break
        return names[:5]
    except Exception:
        # PubChem is supplementary; its failure must not block classification.
        return []


def render_plant_information(plant_data):
    st.markdown(f"### {tr('plant_id')}")
    if plant_data.get("scientific_name"):
        identity_cols = st.columns(2, gap="medium")
        with identity_cols[0]:
            sci = html.escape(str(plant_data["scientific_name"]))
            st.markdown(f'<div class="plant-info-card"><div class="plant-info-title">{tr("scientific")}</div><div class="plant-info-value"><i>{sci}</i></div></div>', unsafe_allow_html=True)
        with identity_cols[1]:
            common = html.escape(str(translate_botanical_text(plant_data.get("common_name") or "—", LANG)))
            st.markdown(f'<div class="plant-info-card"><div class="plant-info-title">{tr("common")}</div><div class="plant-info-value">{common}</div></div>', unsafe_allow_html=True)
        confidence = float(plant_data.get("confidence", 0))
        st.markdown(f'<div class="plant-info-card"><div class="plant-info-title">{tr("confidence")}</div><div class="plant-info-value">{confidence:.1f}%</div></div>', unsafe_allow_html=True)
    else:
        st.warning(tr("not_found"))

    row = plant_data.get("database_row")
    if row:
        st.markdown(f"### {tr('plant_info')}")
        fields = [("Family", "family"), ("Medicinal Rating", "rating"), ("Native Range", "range"), ("Known Hazards", "hazards")]
        info_cols = st.columns(2, gap="medium")
        for index, (column_name, label_key) in enumerate(fields):
            value = row.get(column_name, "—")
            if pd.isna(value) or str(value).strip().lower() in ("", "nan", "none"):
                value = "—"
            if column_name != "Medicinal Rating":
                value = translate_botanical_text(value, LANG)
            safe_value = html.escape(str(value)[:300])
            with info_cols[index % 2]:
                st.markdown(f'<div class="plant-info-card"><div class="plant-info-title">{tr(label_key)}</div><div class="plant-info-value">{safe_value}</div></div>', unsafe_allow_html=True)
        for column_name, label_key, limit in [("Medicinal Properties", "properties", 500), ("Summary", "summary", 500)]:
            value = row.get(column_name, "")
            if not pd.isna(value) and str(value).strip().lower() not in ("", "nan"):
                value = translate_botanical_text(value, LANG)
                safe_value = html.escape(str(value)[:limit])
                st.markdown(f'<div class="plant-info-card"><div class="plant-info-title">{tr(label_key)}</div><div class="plant-info-value">{safe_value}</div></div>', unsafe_allow_html=True)
    elif plant_data.get("scientific_name"):
        st.info(tr("not_db"))

    chemicals = plant_data.get("chemicals") or []
    if chemicals:
        st.markdown(f"### {tr('chemicals')}")
        pills = " ".join(f'<span class="chemical-pill">{html.escape(str(name))}</span>' for name in chemicals)
        st.markdown(pills, unsafe_allow_html=True)
    elif plant_data.get("scientific_name"):
        st.caption(tr("no_chem"))

@st.cache_resource
def load_models():
    m1 = keras.models.load_model("plant_model_v1.keras")
    m2 = keras.models.load_model("plant_model_v2.keras")
    m4 = keras.models.load_model("plant_model_v4.keras")
    return m1, m2, m4

m1, m2, m4 = load_models()
df_plants = load_plant_database()

def predict(model, img):
    x = np.expand_dims(
        np.array(img.convert("RGB").resize((224, 224)), dtype="float32"), axis=0)
    p_non = float(model.predict(x, verbose=0)[0][0])
    return 1 - p_non

def label_v1v2(p_med):
    if p_med > 0.65:
        return tr("med_result"), p_med, "green-fill", "#4ade80"
    if p_med < 0.35:
        return tr("nonmed_result"), 1 - p_med, "red-fill", "#f87171"
    return tr("uncertain"), max(p_med, 1 - p_med), "amber-fill", "#fbbf24"

def label_v4(p_med):
    if p_med >= 0.5:
        return tr("important"), p_med, "green-fill", "#4ade80"
    else:
        return tr("almost"), 1 - p_med, "red-fill", "#f87171"

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

for k in ("results", "feedback_sent", "img_bytes", "plant_data"):
    if k not in st.session_state:
        st.session_state[k] = None

st.markdown(f"""
<div class="hero-box" dir="{'rtl' if LANG == 'ar' else 'ltr'}">
  <h1 style="margin:0 0 10px 0;font-size:2.3rem;font-weight:800;color:white;text-shadow:0 2px 10px rgba(0,0,0,0.25);">🌿 {tr('hero')}</h1>
  <p style="margin:0 0 14px 0;font-size:1.1rem;color:#ecfdf5;opacity:.95;">{tr('subtitle')}</p>
  <span class="hero-badge">{tr('developed')} <b>Hamza Oulbayyad</b> &bull; <b>Rida Bargach</b></span>
</div>
""", unsafe_allow_html=True)

col_results, col_input = st.columns([2.2, 1], gap="large")

with col_input:
    st.markdown(f"#### {tr('input')}")
    tab_up, tab_cam = st.tabs([tr("upload_tab"), tr("camera_tab")])
    with tab_up:
        uploaded = st.file_uploader(tr("upload"), type=["jpg", "jpeg", "png", "webp", "jfif"],
                                    label_visibility="collapsed")
    with tab_cam:
        camera = st.camera_input(tr("capture"), label_visibility="collapsed")
    img_source = uploaded or camera

    if img_source:
        image = Image.open(img_source)
        image = ImageOps.exif_transpose(image)
        prev_buf = _io.BytesIO()
        image.convert("RGB").save(prev_buf, format="JPEG", quality=90)
        prev_b64 = base64.b64encode(prev_buf.getvalue()).decode("utf-8")
        st.markdown(f"""
        <div class="preview-wrapper">
          <div class="ready-pill">{tr("ready_img")}</div>
          <img src="data:image/jpeg;base64,{prev_b64}" alt="Plant Preview" />
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
    if st.button(tr("analyse"), key="analyse_btn", width="stretch"):
        if img_source is None:
            st.warning(tr("upload_first"))
        else:
            image = Image.open(img_source)
            image = ImageOps.exif_transpose(image)
            buf = _io.BytesIO()
            image.convert("RGB").save(buf, format="JPEG", quality=90)
            img_bytes = buf.getvalue()
            st.session_state.img_bytes = img_bytes

            # Plant identification and reference information are supplementary.
            plant_data = {
                "scientific_name": None,
                "common_name": None,
                "confidence": 0.0,
                "database_row": None,
                "chemicals": [],
                "plantnet_error": None,
            }
            with st.spinner(tr("identifying")):
                try:
                    scientific_name, common_name, confidence = identify_plant(img_bytes)
                    plant_data.update({
                        "scientific_name": scientific_name,
                        "common_name": common_name,
                        "confidence": confidence,
                    })
                except Exception as error:
                    plant_data["plantnet_error"] = str(error)

            if plant_data["scientific_name"]:
                plant_data["database_row"] = find_plant_in_database(
                    plant_data["scientific_name"]
                )
                with st.spinner(tr("fetching")):
                    plant_data["chemicals"] = get_chemical_compounds(
                        plant_data["scientific_name"]
                    )

            st.session_state.plant_data = plant_data

            with st.spinner(tr("running")):
                p1 = predict(m1, image)
                p2 = predict(m2, image)
                p4 = predict(m4, image)
                st.session_state.results = (p1, p2, p4)
                st.session_state.feedback_sent = False

with col_results:
    if st.session_state.results:
        if st.session_state.plant_data:
            render_plant_information(st.session_state.plant_data)
            if st.session_state.plant_data.get("plantnet_error"):
                st.warning(
                    tr("plantnet_error")
                    + str(st.session_state.plant_data["plantnet_error"])
                )
            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        p1, p2, p4 = st.session_state.results
        c1, c2, c3 = st.columns(3, gap="medium")

        models_info = [
            (c1, tr("model1"), tr("baseline"), tr("basev1"), p1, label_v1v2, False),
            (c2, tr("model2"), tr("intermediate"), tr("midv2"), p2, label_v1v2, False),
            (c3, tr("model4"), tr("strongweak"), tr("primary"), p4, label_v4, True),
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
            st.markdown(f"""
            <div class="feedback-box" dir="{'rtl' if LANG == 'ar' else 'ltr'}">
              <div style="font-size:1.05rem;font-weight:700;color:#4ade80;margin-bottom:4px;">{tr('evaluate')}</div>
              <div style="font-size:0.92rem;color:#cbd5e1;margin-bottom:12px;">{tr('evaluate_help')}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            if not st.session_state.feedback_sent:
                fc1, fc2 = st.columns(2, gap="medium")
                with fc1:
                    if st.button(tr("yes"), width="stretch"):
                        st.session_state.feedback_sent = "correct"
                        st.rerun()
                with fc2:
                    if st.button(tr("no"), width="stretch", key="wrong_btn_trigger"):
                        st.session_state.feedback_sent = "wrong"
                        st.rerun()

            elif st.session_state.feedback_sent == "correct":
                st.success(tr("thanks"))

            elif st.session_state.feedback_sent == "wrong":
                st.markdown(f"**{tr('select_true')}**")
                fix = st.selectbox(tr("correct_label"), ["", tr("medicinal"), tr("nonmed")],
                                   label_visibility="collapsed")
                if fix and st.button(tr("submit"), width="stretch"):
                    p1, p2, p4 = st.session_state.results
                    lbl4, _, _, _ = label_v4(p4)
                    with st.spinner(tr("saving")):
                        try:
                            fix_for_dataset = "Medicinal" if fix == tr("medicinal") else "Non-medicinal"
                            wrong_for_dataset = (
                                "Strong medical use" if p4 >= 0.5
                                else "Almost no medical use"
                            )
                            save_correction_to_github(
                                wrong_for_dataset, fix_for_dataset, st.session_state.img_bytes)
                            st.session_state.feedback_sent = "done"
                            st.rerun()
                        except Exception as e:
                            st.error(tr("save_error") + str(e))

            elif st.session_state.feedback_sent == "done":
                st.success(tr("saved"))

    else:
        st.markdown(f"""
        <div dir="{'rtl' if LANG == 'ar' else 'ltr'}" style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    height:340px;background:rgba(30,41,59,0.3);border:1px dashed rgba(255,255,255,0.15);
                    border-radius:22px;color:#94a3b8;text-align:center;padding:20px;">
          <div style="font-size:2.5rem;margin-bottom:10px;">🌱</div>
          <div style="font-size:1.1rem;font-weight:600;color:#e2e8f0;">{tr("ready")}</div>
          <div style="font-size:0.95rem;margin-top:4px;">
            {tr("ready_help")}
          </div>
        </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div dir="{'rtl' if LANG == 'ar' else 'ltr'}" style="text-align:center;font-size:.85rem;color:#94a3b8;margin-top:40px;
            border-top:1px solid rgba(255,255,255,0.08);padding-top:18px;">
  {tr("academic")}<br>
  <b>Hamza Oulbayyad & Rida Bargach</b> • {tr("dataset")} • <b>Version 1.1</b>
</div>
""", unsafe_allow_html=True)
