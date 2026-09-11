import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import os

# --- Page configuration ---
st.set_page_config(
    page_title="PneumoScan | AI Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for professional polish ---
st.markdown("""
<style>
    header { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    .result-card {
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #E2E8F0;
        background: white;
        margin-bottom: 1rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0EA5E9;
    }

    .hospital-item {
        padding: 0.75rem;
        border-left: 3px solid #0EA5E9;
        margin-bottom: 0.5rem;
        background: #F8FAFC;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Load model (cached for performance) ---
@st.cache_resource
def load_model():
    # Model file must be in the same folder as app.py on GitHub
    return tf.keras.models.load_model('pneumonia_mobilenetv2.h5')

# --- Grad-CAM heatmap generation ---
def generate_heatmap(model, img_array):
    # Find the last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        # Fallback for nested models (e.g., MobileNetV2 inside Sequential)
        for layer in reversed(model.layers):
            if hasattr(layer, 'layers'):
                for sublayer in reversed(layer.layers):
                    if 'conv' in sublayer.name.lower():
                        last_conv_layer = layer.name
                        break

    if last_conv_layer is None:
        # If no conv layer found, return a blank heatmap
        return np.zeros((7, 7))

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap, alpha=0.4):
    import cv2
    heatmap = cv2.resize(heatmap, (original_img.width, original_img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    original = np.array(original_img)
    overlayed = cv2.addWeighted(original, 1-alpha, heatmap, alpha, 0)
    return Image.fromarray(overlayed)

# --- Free nearby hospital search (OpenStreetMap Nominatim) ---
def search_nearby_hospitals(lat, lon, radius_km=5):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": "hospital",
            "lat": lat,
            "lon": lon,
            "radius": radius_km * 1000,
            "format": "json",
            "limit": 5
        }
        headers = {"User-Agent": "PneumoScan/1.0 (educational project)"}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        return response.json()
    except Exception:
        return []

# --- Main UI ---
st.title("🫁 PneumoScan")
st.caption("AI-powered pneumonia detection with explainable heatmaps and nearby care locator")

# Sidebar for location (for hospital search)
with st.sidebar:
    st.header("📍 Your Location")
    st.caption("Used to find nearby hospitals if pneumonia is detected")
    lat = st.number_input("Latitude", value=28.6139, format="%.4f")
    lng = st.number_input("Longitude", value=77.2090, format="%.4f")
    radius = st.slider("Search radius (km)", 1, 20, 5)
    st.divider()
    st.caption("💡 Tip: Find your coordinates at latlong.net")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload X-Ray")
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

show_hospitals = False

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Uploaded X-Ray", use_container_width=True)

    with col2:
        st.subheader("AI Analysis")

        with st.spinner("Analyzing image..."):
            model = load_model()

            # Preprocess
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = float(model.predict(img_array, verbose=0)[0][0])

            # Generate heatmap
            heatmap = generate_heatmap(model, img_array)
            overlayed = overlay_heatmap(image, heatmap)

        # Results
        if prediction > 0.5:
            st.error("### ⚠️ Pneumonia Detected")
            st.metric("Confidence", f"{prediction:.1%}")
            st.warning("Please consult a healthcare professional. AI results are for screening only.")
            show_hospitals = True
        else:
            st.success("### ✅ No Pneumonia Detected")
            st.metric("Confidence", f"{1-prediction:.1%}")
            st.info("Result appears normal. If symptoms persist, follow medical advice.")
            show_hospitals = False

        # Show Grad-CAM
        st.subheader("Model Attention (Grad-CAM)")
        st.image(overlayed, caption="Red areas indicate regions the model focused on", use_container_width=True)

# Hospital search section
if show_hospitals:
    st.divider()
    st.subheader("🏥 Nearby Healthcare Facilities")
    st.caption(f"Showing hospitals within {radius}km of your location")

    with st.spinner("Searching nearby hospitals..."):
        hospitals = search_nearby_hospitals(lat, lng, radius)

    if hospitals:
        for h in hospitals[:5]:
            name = h.get('display_name', 'Unknown facility')
            short_name = name.split(',')[0] if ',' in name else name
            st.markdown(f"""
            <div class="hospital-item">
                <strong>{short_name}</strong><br>
                <small>{name}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No hospitals found nearby. Try expanding the search radius.")
        st.caption("Note: Data from OpenStreetMap. Coverage varies by region.")

st.divider()
st.caption("⚠️ **Disclaimer**: This tool is for educational screening only. Not a substitute for professional medical diagnosis.")
