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

# -----------------------------
# إعدادات عامة
# -----------------------------
st.set_page_config(page_title="نظام اكتشاف حالات الفاقد الكهربائي للفئة الزراعية", layout="wide")

# إعدادات API للقمر الصناعي
API_KEY = "YOUR_API_KEY"
ZOOM = 15
IMG_SIZE = 640
MAP_TYPE = "satellite"

# المسارات
IMG_DIR = "images"
DETECTED_DIR = "DETECTED_FIELDS/FIELDS/farms"
MODEL_PATH = "best.pt"
ML_MODEL_PATH = "isolation_forest_model.joblib"
SCALER_PATH = "scaler.joblib"
OUTPUT_EXCEL = "output/detected_low_usage.xlsx"

# حدود استهلاك الطاقة
capacity_thresholds = {20: 6000, 50: 15000, 70: 21000, 100: 30000, 150: 45000,
                      200: 60000, 300: 90000, 400: 120000, 500: 150000}

# إنشاء المجلدات
Path(IMG_DIR).mkdir(parents=True, exist_ok=True)
Path(DETECTED_DIR).mkdir(parents=True, exist_ok=True)
Path("output").mkdir(parents=True, exist_ok=True)

# -----------------------------
# تحميل النماذج مرة واحدة
# -----------------------------
@st.cache_resource
def load_models():
    model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
    model_ml = joblib.load(ML_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model_yolo, model_ml, scaler

model_yolo, model_ml, scaler = load_models()

# -----------------------------
# الدوال المساعدة
# -----------------------------
def download_image(lat, lon, meter_id):
    img_path = os.path.join(IMG_DIR, f"{meter_id}.png")
    if os.path.exists(img_path):
        return img_path
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {"center": f"{lat},{lon}", "zoom": ZOOM, "size": f"{IMG_SIZE}x{IMG_SIZE}", "maptype": MAP_TYPE, "key": API_KEY}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        with open(img_path, 'wb') as f:
            f.write(response.content)
        return img_path
    return None

def pixel_to_area(lat, box):
    scale = 156543.03392 * abs(math.cos(math.radians(lat))) / (2 ** ZOOM)
    width_m = abs(box[2] - box[0]) * scale
    height_m = abs(box[3] - box[1]) * scale
    return width_m * height_m

def detect_field(img_path, meter_id, info, model):
    results = model(img_path)
    df_result = results.pandas().xyxy[0]
    fields = df_result[df_result["name"] == "field"]
    if not fields.empty:
        confidence = round(fields["confidence"].max() * 100, 2)
        if confidence >= 85:
            image = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            largest_field = fields.iloc[0]
            box = [largest_field["xmin"], largest_field["ymin"], largest_field["xmax"], largest_field["ymax"]]
            draw.rectangle(box, outline="green", width=3)
            area = pixel_to_area(info['y'], box)
            draw.text((10, 10), f"ID: {meter_id}\nArea: {int(area)} m²", fill="yellow")
            image_path = os.path.join(DETECTED_DIR, f"{meter_id}.png")
            image.save(image_path)
            return confidence, image_path, int(area)
    return None, None, None

def determine_priority(has_field, anomaly, consumption_check, high_priority_condition):
    if high_priority_condition:
        return "أولوية عالية جدًا"
    elif has_field and anomaly and consumption_check:
        return "قصوى"
    elif has_field and (anomaly or consumption_check):
        return "متوسطة"
    elif has_field:
        return "منخفضة"
    return "طبيعية"

def predict_loss(info, model_ml, scaler):
    X = [[info["Breaker Capacity"], info["الكمية"]]]
    X_scaled = scaler.transform(X)
    return model_ml.predict(X_scaled)[0]

def generate_whatsapp_share_link(meter_id, confidence, area, location_link, quantity, capacity, office_number, priority):
    message = f"حالة عداد {meter_id}:\nالمكتب: {office_number}\nأولوية: {priority}\nثقة: {confidence}%\nمساحة: {area} م²\nالاستهلاك: {quantity}\nسعة القاطع: {capacity}\nالموقع: {location_link}"
    return f"https://wa.me/?text={urllib.parse.quote(message)}"

def generate_google_maps_link(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"

# -----------------------------
# واجهة Streamlit
# -----------------------------
st.title("🔍 نظام اكتشاف حالات الفاقد الكهربائي للفئة الزراعية")

uploaded_file = st.file_uploader("📁 ارفع ملف البيانات (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    required_cols = ["Breaker Capacity", "الكمية", "المكتب", "y", "x", "الاشتراك"]
    if not all(col in df.columns for col in required_cols):
        st.error("بعض الأعمدة الأساسية غير موجودة.")
        st.stop()

    results = []
    gallery = []
    progress = st.progress(0)

    for idx, row in df.iterrows():
        meter_id, lat, lon = row['الاشتراك'], row['y'], row['x']
        img_path = download_image(lat, lon, meter_id)
        if img_path:
            conf, img_detected, area = detect_field(img_path, meter_id, row, model_yolo)
            if conf:
                anomaly = predict_loss(row, model_ml, scaler)
                limit = capacity_thresholds.get(row['Breaker Capacity'], 0)
                consumption_check = row['الكمية'] < 0.5 * limit
                high_priority_condition = (conf >= 85 and row['الكمية'] == 0) or (row['Breaker Capacity'] < 200)
                priority = determine_priority(conf >= 85, anomaly, consumption_check, high_priority_condition)
                results.append({**row, "الثقة": conf, "المساحة": area, "الأولوية": priority})
                gallery.append((conf, priority, img_detected, meter_id, area, lat, lon, row['الكمية'], row['Breaker Capacity'], row['المكتب']))

        progress.progress((idx+1)/len(df))

    df_final = pd.DataFrame(results)
    df_final.to_excel(OUTPUT_EXCEL, index=False)
    st.download_button("📥 تحميل النتائج", data=open(OUTPUT_EXCEL, "rb"), file_name="results.xlsx")
