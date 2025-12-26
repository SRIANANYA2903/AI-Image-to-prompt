import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="AI Vision Prompt Pro", page_icon="🎨", layout="wide")

# 2. UI Fix: Full-Width Boxes & High Visibility Text
st.markdown("""
    <style>
    /* Full Page Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #a855f7 0%, #ec4899 50%, #facc15 100%);
        background-attachment: fixed;
    }

    /* Professional Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e1b4b;
    }

    /* Card Styling: Ensuring Full Width and No Gaps */
    .stColumn > div {
        background-color: rgba(255, 255, 255, 0.98);
        padding: 40px !important;
        border-radius: 20px;
        box-shadow: 0px 20px 40px rgba(0,0,0,0.15);
        min-height: 600px; /* Full box feel */
    }

    /* Heading Colors */
    .main-title {
        color: white !important;
        font-size: 50px !important;
        font-weight: 800 !important;
        text-align: center;
        margin-top: -50px;
    }
    
    .card-header {
        color: #1e1b4b !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #ec4899;
        padding-bottom: 10px;
        margin-bottom: 25px;
    }

    /* Prompt Box Styling */
    .prompt-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 10px solid #ec4899;
        padding: 20px;
        border-radius: 12px;
        font-size: 16px;
        color: #1e293b;
        line-height: 1.8;
    }

    /* Green Generate Button */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 4em;
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        font-weight: bold;
        font-size: 18px;
        border: none;
        box-shadow: 0px 10px 20px rgba(16, 185, 129, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Model Loading Logic (CPU/GPU Detection)
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model, device

# 4. Sidebar Content
with st.sidebar:
    st.markdown("<h2 style='color:white;'>👤 Project Dashboard</h2>", unsafe_allow_html=True)
    st.info("**Model:** BLIP Vision AI")
    st.info("**Category:** Image-to-Prompt")
    st.divider()
    st.caption("Developed by SRI ANANYA")

# 5. Header Section
st.markdown('<p class="main-title">✨ AI Image to Ultra-Detailed Prompt</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>Transform images into high-detail professional prompts.</p>", unsafe_allow_html=True)

# 6. Main Grid (Fixed Full-Width Columns)
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown('<p class="card-header">📷 Input Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)

with col2:
    st.markdown('<p class="card-header">🤖 AI Generation Results</p>', unsafe_allow_html=True)
    if uploaded_file:
        if st.button('🚀 GENERATE ULTRA-DETAILED PROMPT'):
            with st.spinner('AI analyzing every detail...'):
                processor, model, device = load_model()
                inputs = processor(image, return_tensors="pt").to(device)
                out = model.generate(**inputs, max_new_tokens=150) # Increased for more detail
                base_caption = processor.decode(out[0], skip_special_tokens=True)

                # --- HYPER-DETAILED PROMPT LOGIC ---
                detailed_prompt = (
                    f"**Ultra-detailed professional photography of {base_caption}.** "
                    "Every surface is rendered with incredibly fine tactile textures. "
                    "Cinematic volumetric atmospheric lighting, high-end 85mm prime lens, f/1.8, ISO 100. "
                    "Shot at the golden hour with global illumination, ray tracing, and rule of thirds. "
                    "Masterpiece quality, sharp focus, Unreal Engine 5 render style, "
                    "8k resolution, photorealistic, subsurface scattering, intricate details."
                )

                st.markdown("##### 🔥 Masterpiece Prompt")
                st.markdown(f'<div class="prompt-box">{detailed_prompt}</div>', unsafe_allow_html=True)
                
                st.markdown("##### 🚫 Negative Prompt")
                st.warning("blurry, low quality, distorted, grainy, low resolution, out of frame")
    else:
        st.info("Waiting for image upload to begin analysis...")

st.markdown("<br><center style='color: white;'>Developed by SRI ANANYA | Professional AI Vision System</center>", unsafe_allow_html=True)
