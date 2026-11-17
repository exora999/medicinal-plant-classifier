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
        "subtitle": "Hybrid Deep Learning & Botanical Knowledge System — powered by <b>Model 4</b>",
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
        "running": "Running Model 4 inference...",
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
        "strongweak": "Hybrid Vision + PFAF Decision",
        "primary": "⭐ Primary Model",
        "source_hybrid": "🧬 Vision + PFAF ({rating}/5)",
        "source_vision": "👁️ Visual CNN Only",
        "evaluate": "🎯 Evaluate Primary Model (Model 4)",
        "evaluate_help": "Is <b>Model 4's</b> classification accurate for this plant?",
        "yes": "✅ Yes, Model 4 is correct",
        "no": "❌ No, Model 4 is wrong",
        "thanks": "Thank you for validating Model 4's prediction! 🎉",
        "select_true": "Select the true classification for this plant:",
        "correct_label": "Correct label",
        "medicinal": "Medicinal",
        "nonmed": "Non-medicinal",
        "submit": "📨 Submit Correction for Model 4",
        "saving": "Saving correction to dataset...",
        "saved": "Correction saved! Thank you for helping improve Model 4 🎉",
        "save_error": "Error saving feedback: ",
        "ready": "Ready to Analyse",
        "ready_help": "Upload a plant photo on the side and click <b>Analyse Plant</b>",
        "academic": "⚠️ <b>Academic & Educational Project</b> — Do NOT use this tool for medical diagnosis or consumption decisions.",
        "dataset": "Dataset: Plants For A Future (pfaf.org)",
        "model4": "Model 4",
        "important": "🟢 Significant medical use",
        "almost": "🔴 Almost no medical use"
    },
    "fr": {
        "lang_label": "Langue d'affichage",
        "hero": "Classification des plantes médicinales",
        "subtitle": "Système hybride Deep Learning & Base botanique — propulsé par le <b>Modèle 4</b>",
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
        "running": "Exécution de la prédiction du Modèle 4...",
        "plant_id": "Identification de la plante",
        "scientific": "Nom scientifique",
        "common": "Nom commun",
        "confidence": "Confiance PlantNet",
        "not_found": "PlantNet n’a pas pu identifier cette plante. Essayez une photo plus nette.",
        "plant_info": "Informations sur la plante (PFAF)",
        "family": "🌱 Famille",
        "rating": "⭐ Note médicinale (0–5)",
        "range": "🗺️ Répartition naturelle",
        "hazards": "⚠️ Dangers connus",
        "properties": "💊 Propriétés médicinales",
        "summary": "📝 Résumé",
        "not_db": "La plante a été identifiée par PlantNet, mais aucune entrée correspondante n’a été trouvée dans pfaf_plants_small.csv.",
        "csv_missing": "Le fichier CSV pfaf_plants_small.csv est introuvable à côté de l'application.",
        "chemicals": "Composés chimiques (PubChem)",
        "chem_note": "Les noms des composés chimiques sont conservés sous leur forme scientifique normalisée.",
        "no_chem": "Aucun nom de composé chimique n’a été fourni par PubChem.",
        "plantnet_error": "L’identification PlantNet est indisponible : ",
        "strongweak": "Décision Hybride Vision + PFAF",
        "primary": "⭐ Modèle principal",
        "source_hybrid": "🧬 Vision + PFAF ({rating}/5)",
        "source_vision": "👁️ Vision CNN uniquement",
        "evaluate": "🎯 Évaluer le modèle principal (Modèle 4)",
        "evaluate_help": "La classification du <b>Modèle 4</b> est-elle correcte pour cette plante ?",
        "yes": "✅ Oui, le Modèle 4 est correct",
        "no": "❌ Non, le Modèle 4 est incorrect",
        "thanks": "Merci d’avoir validé la prédiction du Modèle 4 ! 🎉",
        "select_true": "Sélectionnez la classification réelle de cette plante :",
        "correct_label": "Étiquette correcte",
        "medicinal": "Médicinale",
        "nonmed": "Non médicinale",
        "submit": "📨 Envoyer la correction du Modèle 4",
        "saving": "Enregistrement de la correction...",
        "saved": "Correction enregistrée ! Merci de contribuer à améliorer le Modèle 4 🎉",
        "save_error": "Erreur lors de l’enregistrement : ",
        "ready": "Prêt à analyser",
        "ready_help": "Importez une photo de plante puis cliquez sur <b>Analyser la plante</b>",
        "academic": "⚠️ <b>Projet académique et éducatif</b> — N’utilisez PAS cet outil pour un diagnostic médical ou une décision de consommation.",
        "dataset": "Base de données : Plants For A Future (pfaf.org)",
        "model4": "Modèle 4",
        "important": "🟢 Usage médicinal important",
        "almost": "🔴 Usage médicinal quasi inexistant"
    },
    "ar": {
        "lang_label": "لغة العرض",
        "hero": "تصنيف النباتات الطبية",
        "subtitle": "نظام هجين يجمع بين التعلم العميق وقاعدة بيانات PFAF — بالاعتماد على <b>النموذج 4</b>",
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
        "running": "جارٍ تشغيل النموذج 4...",
        "plant_id": "التعرف على النبتة",
        "scientific": "الاسم العلمي",
        "common": "الاسم الشائع",
        "confidence": "نسبة ثقة PlantNet",
        "not_found": "لم يتمكن PlantNet من تحديد النبتة. جرّب صورة أوضح.",
        "plant_info": "معلومات النبتة (PFAF)",
        "family": "🌱 الفصيلة",
        "rating": "⭐ التقييم الطبي (0–5)",
        "range": "🗺️ الموطن الأصلي",
        "hazards": "⚠️ المخاطر المعروفة",
        "properties": "💊 الخصائص الطبية",
        "summary": "📝 ملخص",
        "not_db": "تم التعرف على النبتة بواسطة PlantNet، لكن لم يُعثر على معلومات مطابقة في قاعدة البيانات.",
        "csv_missing": "ملف قاعدة البيانات pfaf_plants_small.csv غير موجود بجانب التطبيق.",
        "chemicals": "المركبات الكيميائية (PubChem)",
        "chem_note": "تُترك أسماء المركبات الكيميائية بصيغتها العلمية القياسية حفاظاً على دقتها.",
        "no_chem": "لم يُرجع PubChem أسماء للمركبات الكيميائية.",
        "plantnet_error": "خدمة PlantNet غير متاحة حالياً: ",
        "strongweak": "قرار هجين: الرؤية البصرية + PFAF",
        "primary": "⭐ النموذج الرئيسي",
        "source_hybrid": "🧬 الرؤية البصرية + PFAF ({rating}/5)",
        "source_vision": "👁️ الرؤية البصرية فقط",
        "evaluate": "🎯 تقييم النموذج الرئيسي (النموذج 4)",
        "evaluate_help": "هل تصنيف <b>النموذج 4</b> صحيح لهذه النبتة؟",
        "yes": "✅ نعم، تصنيف النموذج 4 صحيح",
        "no": "❌ لا، تصنيف النموذج 4 غير صحيح",
        "thanks": "شكراً لتأكيد نتيجة النموذج 4! 🎉",
        "select_true": "اختر التصنيف الصحيح لهذه النبتة:",
        "correct_label": "التصنيف الصحيح",
        "medicinal": "طبية",
        "nonmed": "غير طبية",
        "submit": "📨 إرسال تصحيح النموذج 4",
        "saving": "جارٍ حفظ التصحيح...",
        "saved": "تم حفظ التصحيح! شكراً لمساهمتك في تحسين النموذج 4 🎉",
        "save_error": "خطأ أثناء حفظ الملاحظة: ",
        "ready": "جاهز للتحليل",
        "ready_help": "ارفع صورة النبتة من الجهة الجانبية ثم اضغط على <b>تحليل النبتة</b>",
        "academic": "⚠️ <b>مشروع أكاديمي وتعليمي</b> — لا تستخدم هذه الأداة للتشخيص الطبي أو لاتخاذ قرارات الاستهلاك.",
        "dataset": "قاعدة البيانات: Plants For A Future (pfaf.org)",
        "model4": "النموذج 4",
        "important": "🟢 استخدام طبي مهم",
        "almost": "🔴 استخدام طبي شبه منعدم"
    }
}

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Cairo:wght@500;600;700;800&display=swap');

  html, body, [class*="css"] {
      font-family: 'Plus Jakarta Sans', 'Cairo', sans-serif;
  }
  header[data-testid="stHeader"] {
      background: transparent !important;
  }
  .block-container {
      padding-top: 2.2rem !important;
  }
  .stApp {
      background: radial-gradient(circle at 50% -10%, #064e3b 0%, #0f172a 55%, #020617 100%);
      color: #f8fafc;
  }
  .top-lang-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(6, 78, 59, 0.45);
      border: 1px solid rgba(74, 222, 128, 0.3);
      border-radius: 999px;
      padding: 7px 16px;
      font-size: 0.82rem;
      font-weight: 700;
      color: #a7f3d0;
      backdrop-filter: blur(10px);
      margin-top: 4px;
  }
  div[data-testid="stSelectbox"] > label {
      display: none !important;
  }
  div[data-testid="stSelectbox"] > div > div {
      background: linear-gradient(135deg, rgba(6, 78, 59, 0.65), rgba(15, 23, 42, 0.85)) !important;
      border: 1px solid rgba(74, 222, 128, 0.42) !important;
      border-radius: 14px !important;
      color: #f8fafc !important;
      font-weight: 700 !important;
      backdrop-filter: blur(12px);
      box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
      transition: all 0.25s ease !important;
  }
  div[data-testid="stSelectbox"] > div > div:hover {
      border-color: #4ade80 !important;
      box-shadow: 0 0 16px rgba(74, 222, 128, 0.3);
  }
  div[data-testid="stSelectbox"] svg {
      fill: #4ade80 !important;
  }
  .hero-box {
      background: linear-gradient(135deg, rgba(6, 78, 59, 0.85), rgba(22, 163, 74, 0.75), rgba(16, 185, 129, 0.65));
      backdrop-filter: blur(12px);
      border: 1px solid rgba(74, 222, 128, 0.35);
      border-radius: 26px;
      padding: 34px 24px;
      text-align: center;
      color: white;
      margin-top: 8px;
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
  .model-card-primary {
      background: linear-gradient(180deg, rgba(6, 78, 59, 0.48) 0%, rgba(15, 23, 42, 0.85) 100%);
      backdrop-filter: blur(12px);
      border: 1.5px solid #4ade80;
      border-radius: 22px;
      padding: 28px 24px;
      text-align: center;
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
      font-size: 0.78rem;
      font-weight: 700;
      padding: 4px 14px;
      border-radius: 999px;
      margin-bottom: 10px;
      letter-spacing: 0.5px;
      text-transform: uppercase;
  }
  .source-pill-hybrid {
      display: inline-block;
      margin-top: 14px;
      background: rgba(16, 185, 129, 0.18);
      border: 1px solid rgba(52, 211, 153, 0.45);
      color: #6ee7b7;
      font-size: 0.8rem;
      font-weight: 700;
      padding: 5px 14px;
      border-radius: 999px;
  }
  .source-pill-vision {
      display: inline-block;
      margin-top: 14px;
      background: rgba(148, 163, 184, 0.14);
      border: 1px solid rgba(148, 163, 184, 0.3);
      color: #cbd5e1;
      font-size: 0.8rem;
      font-weight: 700;
      padding: 5px 14px;
      border-radius: 999px;
  }
  .model-title {
      font-size: 1.45rem;
      font-weight: 800;
      color: #f8fafc;
      margin-bottom: 4px;
  }
  .model-sub {
      font-size: 0.9rem;
      color: #94a3b8;
      margin-bottom: 20px;
  }
  .bar-wrap {
      max-width: 540px;
      margin: 10px auto 6px auto;
  }
  .bar-label {
      font-size: 1.08rem;
      font-weight: 700;
      margin-bottom: 10px;
  }
  .bar-track {
      background: rgba(15, 23, 42, 0.85);
      border-radius: 999px;
      height: 26px;
      width: 100%;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.08);
      padding: 2px;
  }
  .bar-fill {
      height: 20px;
      border-radius: 999px;
      display: flex;
      align-items: center;
      padding-right: 12px;
      justify-content: flex-end;
      color: white;
      font-size: 0.86rem;
      font-weight: 800;
      min-width: 54px;
      transition: width 0.6s ease;
  }
  .green-fill { background: linear-gradient(90deg, #15803d, #22c55e, #4ade80); }
  .red-fill   { background: linear-gradient(90deg, #991b1b, #ef4444, #f87171); }
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
  .plant-section-heading {
      display: flex;
      align-items: center;
      gap: 11px;
      margin: 20px 0 12px 0;
      padding: 12px 16px;
      border-radius: 15px;
      background: linear-gradient(100deg, rgba(6, 78, 59, .62), rgba(30, 41, 59, .52));
      border: 1px solid rgba(74, 222, 128, .28);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.05);
      color: #ecfdf5;
      font-size: 1.08rem;
      font-weight: 800;
  }
  .plant-section-heading .section-mark {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 30px;
      height: 30px;
      border-radius: 10px;
      background: rgba(74, 222, 128, .16);
      border: 1px solid rgba(74, 222, 128, .32);
      color: #86efac;
      flex: 0 0 30px;
  }
  .plant-info-card {
      position: relative;
      min-height: 92px;
      height: calc(100% - 12px);
      background: linear-gradient(135deg, rgba(6, 78, 59, .88) 0%, rgba(13, 90, 82, .84) 52%, rgba(20, 48, 62, .92) 100%);
      border: 1px solid rgba(74, 222, 128, .38);
      border-radius: 17px;
      padding: 16px 19px 17px 21px;
      margin: 7px 0 13px 0;
      box-shadow: 0 10px 26px rgba(0, 0, 0, .24), inset 0 1px 0 rgba(255,255,255,.10);
      backdrop-filter: blur(14px);
      overflow: hidden;
      transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease;
  }
  .plant-info-card::before {
      content: "";
      position: absolute;
      top: 12px;
      bottom: 12px;
      left: 0;
      width: 3.5px;
      border-radius: 0 5px 5px 0;
      background: linear-gradient(180deg, #4ade80, #0d9488);
  }
  .plant-info-card:hover {
      transform: translateY(-2px);
      border-color: rgba(74, 222, 128, .55);
      box-shadow: 0 13px 28px rgba(0, 0, 0, .28), 0 0 18px rgba(16, 185, 129, .1);
  }
  .plant-info-title {
      display: block;
      font-size: .78rem;
      color: #a7f3d0;
      font-weight: 800;
      margin-bottom: 8px;
      letter-spacing: .03em;
  }
  .plant-info-value {
      font-size: .98rem;
      line-height: 1.75;
      color: #ffffff;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
  }
  .chemical-panel {
      background: linear-gradient(145deg, rgba(6, 78, 59, .3), rgba(15, 23, 42, .65));
      border: 1px solid rgba(74, 222, 128, .26);
      border-radius: 18px;
      padding: 15px 17px;
      margin: 8px 0 16px 0;
  }
  .chemical-pill {
      display: inline-block;
      direction: ltr;
      background: linear-gradient(135deg, rgba(22, 163, 74, .24), rgba(13, 148, 136, .18));
      border: 1px solid rgba(74, 222, 128, .36);
      border-radius: 10px;
      padding: 7px 12px;
      margin: 5px 6px 5px 0;
      color: #bbf7d0;
      font-size: .84rem;
      font-weight: 700;
  }
  [dir="rtl"] .plant-info-card {
      padding: 16px 21px 17px 19px;
      text-align: right;
  }
  [dir="rtl"] .plant-info-card::before {
      left: auto;
      right: 0;
      border-radius: 5px 0 0 5px;
  }
  [dir="rtl"] .chemical-pill {
      margin: 5px 0 5px 6px;
  }
</style>
""", unsafe_allow_html=True)

top_left, top_right = st.columns([3.1, 1.1])
with top_left:
    st.markdown(
        '<div class="top-lang-badge">🌿 Hybrid AI Phytotherapy &bull; PlantNet &bull; PFAF &bull; PubChem</div>',
        unsafe_allow_html=True
    )
with top_right:
    language_label = st.selectbox(
        "Language",
        list(LANGUAGES.keys()),
        index=0,
        key="site_language",
        label_visibility="collapsed"
    )

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
        cut = max(
            remaining.rfind("\n", 0, max_len),
            remaining.rfind(". ", 0, max_len),
            remaining.rfind("; ", 0, max_len),
            remaining.rfind(", ", 0, max_len),
            remaining.rfind(" ", 0, max_len),
        )
        if cut < 200:
            cut = max_len
        else:
            cut += 1
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    return chunks

def _google_translate_chunk(chunk: str, target: str) -> str:
    params = urllib.parse.urlencode({
        "client": "gtx",
        "sl": "auto",
        "tl": target,
        "dt": "t",
        "q": chunk,
    })
    url = f"https://translate.googleapis.com/translate_a/single?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    parts = [seg[0] for seg in payload[0] if seg and seg[0]]
    result = "".join(parts).strip()
    if not result:
        raise RuntimeError("Empty Google translation")
    return result

def _mymemory_translate_chunk(chunk: str, target: str) -> str:
    params = urllib.parse.urlencode({
        "q": chunk[:480],
        "langpair": f"en|{target}",
    })
    url = f"https://api.mymemory.translated.net/get?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    result = payload.get("responseData", {}).get("translatedText", "").strip()
    if not result:
        raise RuntimeError("Empty MyMemory translation")
    return result

@st.cache_data(ttl=86400, show_spinner=False)
def _translate_cached(value: str, target: str) -> str:
    chunks = _split_text_chunks(value, max_len=1300)
    translated_chunks = []
    for chunk in chunks:
        try:
            translated_chunks.append(_google_translate_chunk(chunk, target))
        except Exception:
            sub_chunks = _split_text_chunks(chunk, max_len=450)
            sub_translated = [_mymemory_translate_chunk(sc, target) for sc in sub_chunks]
            translated_chunks.append(" ".join(sub_translated))
    return "\n".join(translated_chunks)

def translate_botanical_text(value, target):
    if value is None:
        return value
    original = str(value).strip()
    if not original or original == "—" or target == "en":
        return value
    try:
        return _translate_cached(original, target)
    except Exception:
        return value

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
        data=json.dumps({
            "message": f"correction (Model 4): {wrong} -> {correct}",
            "content": b64
        }).encode(),
        method="PUT",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json"
        }
    )
    try:
        with urllib.request.urlopen(req) as r:
            img_url = json.loads(r.read())["content"]["html_url"]
    except Exception:
        img_url = None

    body = (
        f"**Model Evaluated:** Model 4 (Hybrid Primary)\n\n"
        f"**Model 4 Prediction:** {wrong}\n\n"
        f"**User Correct Label:** {correct}\n\n"
        + (f"[📷 Plant Image]({img_url})\n\n" if img_url else "")
        + "*Submitted via Medicinal Plant Classifier*"
    )
    req2 = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues",
        data=json.dumps({
            "title": f"Model 4 Correction: {wrong} -> {correct}",
            "body": body,
            "labels": ["correction"]
        }).encode(),
        method="POST",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json"
        }
    )
    with urllib.request.urlopen(req2) as r:
        return json.loads(r.read())["html_url"]

@st.cache_data(show_spinner=False)
def load_plant_database():
    candidates = [
        Path(__file__).resolve().parent / "pfaf_plants_small.csv",
        Path.cwd() / "pfaf_plants_small.csv",
    ]
    for csv_path in candidates:
        if csv_path.is_file():
            try:
                return pd.read_csv(csv_path)
            except Exception:
                return pd.DataFrame()
    return pd.DataFrame()

def identify_plant(img_bytes: bytes, target_language: str = "en"):
    api_key = st.secrets.get("PLANTNET_API_KEY", "")
    if not api_key:
        raise RuntimeError("PLANTNET_API_KEY is missing from Streamlit secrets.")

    boundary = "----PlantNetBoundary" + uuid.uuid4().hex
    query = urllib.parse.urlencode({
        "api-key": api_key,
        "include-related-images": "false",
        "lang": target_language if target_language in ("en", "fr", "ar") else "en",
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
    if not scientific_name or df_plants.empty:
        return None

    def norm(value):
        return "".join(ch.lower() for ch in str(value) if ch.isalnum())

    name_column = None
    for column in df_plants.columns:
        normalized = norm(column)
        if normalized in {
            "scientificname", "botanicalname", "latinname", "species",
            "scientific", "plantlatinname", "taxon"
        }:
            name_column = column
            break
    if name_column is None:
        return None

    names = df_plants[name_column].fillna("").astype(str).str.strip()
    target = norm(scientific_name)
    target_tokens = str(scientific_name).casefold().split()
    target_binomial = " ".join(target_tokens[:2])

    normalized_names = names.map(norm)
    exact = df_plants[normalized_names == target]
    if not exact.empty:
        return exact.iloc[0].to_dict()

    for idx, candidate in names.items():
        candidate_binomial = " ".join(str(candidate).casefold().split()[:2])
        if candidate_binomial == target_binomial:
            return df_plants.loc[idx].to_dict()

    genus = target_tokens[0] if target_tokens else ""
    if genus:
        partial = df_plants[normalized_names.str.startswith(norm(genus))]
        if not partial.empty:
            return partial.iloc[0].to_dict()
    return None

def extract_pfaf_rating(database_row):
    if not database_row:
        return None
    for col, val in database_row.items():
        norm_col = "".join(ch.lower() for ch in str(col) if ch.isalnum())
        if norm_col in {"medicinalrating", "rating", "medrating"}:
            try:
                if val is None or pd.isna(val):
                    return None
                rating = float(val)
                if 0.0 <= rating <= 5.0:
                    return rating
            except (TypeError, ValueError):
                continue
    return None

def compute_hybrid_prediction(p_visual: float, plant_data: dict):
    if not plant_data:
        return p_visual, "vision", None
    confidence = float(plant_data.get("confidence") or 0.0)
    database_row = plant_data.get("database_row")
    rating = extract_pfaf_rating(database_row)

    if rating is not None and confidence >= 35.0:
        rating_prob_map = {
            0: 0.04,
            1: 0.22,
            2: 0.68,
            3: 0.84,
            4: 0.94,
            5: 0.99,
        }
        rounded_rating = int(round(rating))
        p_pfaf = rating_prob_map.get(rounded_rating, min(max(rating / 5.0, 0.01), 0.99))
        alpha = 0.80 if confidence >= 70.0 else (confidence / 100.0)
        p_hybrid = (1.0 - alpha) * p_visual + alpha * p_pfaf
        return float(p_hybrid), "hybrid", rounded_rating

    return p_visual, "vision", None

def get_chemical_compounds(scientific_name: str):
    if not scientific_name:
        return []
    try:
        encoded_name = urllib.parse.quote(scientific_name)
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
            f"name/{encoded_name}/JSON?MaxRecords=5"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
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
        return []

def render_plant_information(plant_data):
    text_dir = "rtl" if LANG == "ar" else "ltr"
    st.markdown(
        f'<div dir="{text_dir}" class="plant-section-heading">'
        f'<span class="section-mark">🌿</span>{html.escape(tr("plant_id"))}</div>',
        unsafe_allow_html=True,
    )

    row = plant_data.get("database_row")

    if plant_data.get("scientific_name"):
        identity_cols = st.columns(2, gap="medium")
        with identity_cols[0]:
            sci = html.escape(str(plant_data["scientific_name"]))
            st.markdown(
                f'<div dir="{text_dir}" class="plant-info-card">'
                f'<div class="plant-info-title">{tr("scientific")}</div>'
                f'<div class="plant-info-value"><i dir="ltr">{sci}</i></div></div>',
                unsafe_allow_html=True,
            )
        with identity_cols[1]:
            raw_common = plant_data.get("common_name") or ""
            if not raw_common and row:
                raw_common = str(row.get("Common Name") or row.get("Plant Name") or "").strip()
            if raw_common and raw_common.lower() not in ("nan", "none", "null"):
                common = html.escape(str(translate_botanical_text(raw_common, LANG)))
            else:
                common = "—"
            st.markdown(
                f'<div dir="{text_dir}" class="plant-info-card">'
                f'<div class="plant-info-title">{tr("common")}</div>'
                f'<div class="plant-info-value">{common}</div></div>',
                unsafe_allow_html=True,
            )
        confidence = float(plant_data.get("confidence", 0))
        st.markdown(
            f'<div dir="{text_dir}" class="plant-info-card">'
            f'<div class="plant-info-title">{tr("confidence")}</div>'
            f'<div class="plant-info-value"><span dir="ltr">{confidence:.1f}%</span></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning(tr("not_found"))

    if row:
        st.markdown(
            f'<div dir="{text_dir}" class="plant-section-heading">'
            f'<span class="section-mark">📋</span>{html.escape(tr("plant_info"))}</div>',
            unsafe_allow_html=True,
        )
        label_map = {
            "Family": "family",
            "Medicinal Rating": "rating",
            "Native Range": "range",
            "Known Hazards": "hazards",
            "Medicinal Properties": "properties",
            "Summary": "summary",
        }
        excluded_columns = {
            "Scientific Name", "Common Name", "Plant Name", "ID", "id",
            "Slug", "URL", "Source", "Image", "Image URL",
        }
        info_cols = st.columns(2, gap="medium")
        card_index = 0

        if LANG != "en":
            with st.spinner(tr("translating")):
                translated_items = []
                for column_name, raw_value in row.items():
                    if column_name in excluded_columns or raw_value is None:
                        continue
                    try:
                        if pd.isna(raw_value):
                            continue
                    except (TypeError, ValueError):
                        pass
                    value_text = str(raw_value).strip()
                    if not value_text or value_text.lower() in ("nan", "none", "null"):
                        continue

                    is_numeric = isinstance(raw_value, (int, float, np.integer, np.floating))
                    if is_numeric:
                        display_val = f"{float(raw_value):.0f} / 5" if column_name == "Medicinal Rating" else str(raw_value)
                    else:
                        display_val = translate_botanical_text(value_text, LANG)

                    label = (
                        tr(label_map[column_name])
                        if column_name in label_map
                        else translate_botanical_text(str(column_name), LANG)
                    )
                    translated_items.append((label, display_val))
        else:
            translated_items = []
            for column_name, raw_value in row.items():
                if column_name in excluded_columns or raw_value is None:
                    continue
                try:
                    if pd.isna(raw_value):
                        continue
                except (TypeError, ValueError):
                    pass
                value_text = str(raw_value).strip()
                if not value_text or value_text.lower() in ("nan", "none", "null"):
                    continue

                is_numeric = isinstance(raw_value, (int, float, np.integer, np.floating))
                if is_numeric:
                    display_val = f"{float(raw_value):.0f} / 5" if column_name == "Medicinal Rating" else str(raw_value)
                else:
                    display_val = value_text

                label = tr(label_map[column_name]) if column_name in label_map else str(column_name)
                translated_items.append((label, display_val))

        for label, display_val in translated_items:
            safe_label = html.escape(str(label))
            safe_value = html.escape(str(display_val)[:2200])
            with info_cols[card_index % 2]:
                st.markdown(
                    f'<div dir="{text_dir}" class="plant-info-card">'
                    f'<div class="plant-info-title">{safe_label}</div>'
                    f'<div class="plant-info-value">{safe_value}</div></div>',
                    unsafe_allow_html=True,
                )
            card_index += 1

    elif plant_data.get("scientific_name"):
        if df_plants.empty:
            st.markdown(
                f'<div dir="{text_dir}" class="plant-info-card">'
                f'<div class="plant-info-title">{html.escape(tr("plant_info"))}</div>'
                f'<div class="plant-info-value">{html.escape(tr("csv_missing"))}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info(tr("not_db"))

    chemicals = plant_data.get("chemicals") or []
    if chemicals:
        st.markdown(
            f'<div dir="{text_dir}" class="plant-section-heading">'
            f'<span class="section-mark">🧪</span>{html.escape(tr("chemicals"))}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div dir="{text_dir}" class="chemical-panel">'
            f'<div style="color:#a7f3d0;font-size:.86rem;margin-bottom:8px;">{html.escape(tr("chem_note"))}</div>'
            + " ".join(
                f'<span dir="ltr" class="chemical-pill">{html.escape(str(name))}</span>'
                for name in chemicals
            )
            + '</div>',
            unsafe_allow_html=True,
        )
    elif plant_data.get("scientific_name"):
        st.caption(tr("no_chem"))

@st.cache_resource
def load_model():
    return keras.models.load_model("plant_model_v4.keras")

m4 = load_model()
df_plants = load_plant_database()

def predict(model, img):
    x = np.expand_dims(
        np.array(img.convert("RGB").resize((224, 224)), dtype="float32"), axis=0
    )
    p_non = float(model.predict(x, verbose=0)[0][0])
    return 1 - p_non

def label_v4(p_med):
    if p_med >= 0.5:
        return tr("important"), p_med, "green-fill", "#4ade80"
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

for k in ("results", "feedback_sent", "img_bytes", "plant_data", "m4_meta"):
    if k not in st.session_state:
        st.session_state[k] = None

page_dir = "rtl" if LANG == "ar" else "ltr"

st.markdown(f"""
<div class="hero-box" dir="{page_dir}">
  <h1 style="margin:0 0 10px 0;font-size:2.3rem;font-weight:800;color:white;text-shadow:0 2px 10px rgba(0,0,0,0.25);">🌿 {tr('hero')}</h1>
  <p style="margin:0 0 14px 0;font-size:1.1rem;color:#ecfdf5;opacity:.95;">{tr('subtitle')}</p>
  <span class="hero-badge">{tr('developed')} <b>Hamza Oulbayyad</b> &bull; <b>Rida Bargach</b></span>
</div>
""", unsafe_allow_html=True)

col_results, col_input = st.columns([2.2, 1], gap="large")

with col_input:
    st.markdown(f'<div dir="{page_dir}"><h4>{tr("input")}</h4></div>', unsafe_allow_html=True)
    tab_up, tab_cam = st.tabs([tr("upload_tab"), tr("camera_tab")])
    with tab_up:
        uploaded = st.file_uploader(
            tr("upload"),
            type=["jpg", "jpeg", "png", "webp", "jfif"],
            label_visibility="collapsed"
        )
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
                    scientific_name, common_name, confidence = identify_plant(img_bytes, LANG)
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
                p4_raw = predict(m4, image)
                p4_final, m4_mode, pfaf_rating = compute_hybrid_prediction(p4_raw, plant_data)
                st.session_state.results = p4_final
                st.session_state.m4_meta = (m4_mode, pfaf_rating)
                st.session_state.feedback_sent = False

with col_results:
    if st.session_state.results is not None:
        if st.session_state.plant_data:
            render_plant_information(st.session_state.plant_data)
            if st.session_state.plant_data.get("plantnet_error"):
                st.warning(
                    tr("plantnet_error")
                    + str(st.session_state.plant_data["plantnet_error"])
                )
            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        p4 = st.session_state.results
        m4_mode, pfaf_rating = st.session_state.m4_meta or ("vision", None)
        lbl, conf, fill_cls, txt_col = label_v4(p4)

        if m4_mode == "hybrid" and pfaf_rating is not None:
            source_txt = tr("source_hybrid").format(rating=pfaf_rating)
            source_html = f'<div class="source-pill-hybrid">{source_txt}</div>'
        else:
            source_html = f'<div class="source-pill-vision">{tr("source_vision")}</div>'

        st.markdown(f"""
        <div class="model-card-primary" dir="{page_dir}">
          <span class="primary-tag">{tr("primary")}</span>
          <div class="model-title">{tr("model4")}</div>
          <div class="model-sub">{tr("strongweak")}</div>
          {bar(lbl, conf, fill_cls, txt_col)}
          {source_html}
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

        with st.container():
            st.markdown(f"""
            <div class="feedback-box" dir="{page_dir}">
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
                st.markdown(f'<div dir="{page_dir}"><b>{tr("select_true")}</b></div>', unsafe_allow_html=True)
                fix = st.selectbox(
                    tr("correct_label"),
                    ["", tr("medicinal"), tr("nonmed")],
                    label_visibility="collapsed"
                )
                if fix and st.button(tr("submit"), width="stretch"):
                    with st.spinner(tr("saving")):
                        try:
                            fix_for_dataset = "Medicinal" if fix == tr("medicinal") else "Non-medicinal"
                            wrong_for_dataset = (
                                "Strong medical use" if p4 >= 0.5
                                else "Almost no medical use"
                            )
                            save_correction_to_github(
                                wrong_for_dataset,
                                fix_for_dataset,
                                st.session_state.img_bytes
                            )
                            st.session_state.feedback_sent = "done"
                            st.rerun()
                        except Exception as e:
                            st.error(tr("save_error") + str(e))

            elif st.session_state.feedback_sent == "done":
                st.success(tr("saved"))

    else:
        st.markdown(f"""
        <div dir="{page_dir}" style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    height:340px;background:rgba(30,41,59,0.3);border:1px dashed rgba(255,255,255,0.15);
                    border-radius:22px;color:#94a3b8;text-align:center;padding:20px;">
          <div style="font-size:2.5rem;margin-bottom:10px;">🌱</div>
          <div style="font-size:1.1rem;font-weight:600;color:#e2e8f0;">{tr("ready")}</div>
          <div style="font-size:0.95rem;margin-top:4px;">
            {tr("ready_help")}
          </div>
        </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div dir="{page_dir}" style="text-align:center;font-size:.85rem;color:#94a3b8;margin-top:40px;
            border-top:1px solid rgba(255,255,255,0.08);padding-top:18px;">
  {tr("academic")}<br>
  <b>Hamza Oulbayyad & Rida Bargach</b> • {tr("dataset")} • <b>Version 1.4</b>
</div>
""", unsafe_allow_html=True)
