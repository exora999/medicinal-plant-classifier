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
      background: rgba(74, 222, 1
