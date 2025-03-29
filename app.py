import os
import math
import pandas as pd
import requests
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import joblib
from sklearn.preprocessing import StandardScaler
import streamlit as st
import urllib.parse
from ultralytics import YOLO

# -------------------------
# إعدادات عامة
# -------------------------
st.set_page_config(page_title="نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية", layout="wide")

# إعدادات API للقمر الصناعي
API_KEY = "AIzaSyAY7NJrBjS42s6upa9z_qgNLVXESuu366Q"
ZOOM = 15
IMG_SIZE = 640
MAP_TYPE = "satellite"

# إعدادات المجلدات والمسارات
IMG_DIR = "images"
DETECTED_DIR = "DETECTED_FIELDS/FIELDS/farms"
MODEL_PATH = "best.pt"
ML_MODEL_PATH = "final_model.joblib"
OUTPUT_EXCEL = "output/detected_low_usage.xlsx"

# تحميل نموذج YOLOv5 باستخدام Ultralytics
try:
    model_yolo = YOLO(MODEL_PATH)
    st.success("✅ تم تحميل نموذج YOLOv5 بنجاح")
except Exception as e:
    st.error(f"❗ خطأ في تحميل نموذج YOLOv5: {e}")

# تحميل نموذج تعلم الآلة
try:
    model_ml = joblib.load(ML_MODEL_PATH)
    st.success("✅ تم تحميل نموذج تعلم الآلة بنجاح")
except Exception as e:
    st.error(f"❗ خطأ في تحميل نموذج تعلم الآلة: {e}")

# تحميل نموذج البيانات المطلوب تحليله
st.subheader("📥 تحميل نموذج البيانات المطلوب تحليله")
template_file = "fram.xlsx"
st.download_button("📥 تحميل نموذج البيانات", open(template_file, "rb"), file_name=template_file)

capacity_thresholds = {
    20: 6000, 50: 15000, 70: 21000, 100: 30000, 150: 45000,
    200: 60000, 300: 90000, 400: 120000, 500: 150000
}

Path(IMG_DIR).mkdir(parents=True, exist_ok=True)
Path(DETECTED_DIR).mkdir(parents=True, exist_ok=True)
Path("output").mkdir(parents=True, exist_ok=True)

st.title("🔍 نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية")

uploaded_file = st.file_uploader("📁 ارفع ملف البيانات (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df["cont"] = df["الاشتراك"].astype(str).str.strip()
    df["المكتب"] = df["المكتب"].astype(str)
    df["الكمية"] = pd.to_numeric(df["الكمية"], errors="coerce")

    st.success("✅ تم رفع الملف بنجاح")

    # تحليل البيانات
    results = []
    for _, row in df.iterrows():
        if row['cont'] and row['المكتب'] and not pd.isnull(row['الكمية']):
            prediction = model_ml.predict([[row['الكمية']]])[0]
            results.append({
                'الاشتراك': row['cont'],
                'المكتب': row['المكتب'],
                'الكمية': row['الكمية'],
                'التوقع': prediction
            })
    result_df = pd.DataFrame(results)
    st.write("📊 نتائج التوقع:")
    st.write(result_df)

    # تحميل النتائج
    result_file = 'output/results.xlsx'
    result_df.to_excel(result_file, index=False)
    with open(result_file, "rb") as f:
        st.download_button("📥 تحميل نتائج التوقع", f, file_name="results.xlsx")

else:
    st.warning("⚠️ يرجى رفع ملف البيانات للمتابعة")
