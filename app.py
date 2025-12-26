import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="AI Vision Prompt Pro", page_icon="🎨", layout="wide")

# 2. Advanced Professional CSS (Matching the UI Image)
st.markdown("""
    <style>
    /* Main Background and Fonts */
    .main { background-color: #f0f2f6; color: #1e1e1e; font-family: 'Inter', sans-serif; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { background-color: #1e1e2f; color: white; }
    
    /* Card-like Containers */
    .stColumn {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* Header Styling */
    h1 { color: #1e1e2f; font-weight: 800; letter-spacing: -1px; }
    
    /* Output Box Styling (Professional White with Border) */
    .prompt-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #f8f9fa;
        color: #2c3e50;
        border: 1px dashed #2e7d32;
        font-size: 15px;
        line-height: 1.6;
        margin-top: 10px;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3.5em;
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        color: white;
        font-weight: 600;
        border: none;
        box-shadow: 0px 4px 10px rgba(46, 125, 50, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Model Loading (Safe for Streamlit Cloud)
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model, device

# 4. Sidebar Content
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.title("Project Dashboard")
    st.markdown("---")
    st.write("📂 **Model:** BLIP Vision")
    st.write("🎯 **Task:** Hyper-Detail Gen")
    st.write("⚡ **Status:** Ready")
    st.divider()
    st.caption("AI-Powered Prompt Engineering System")

# 5. Main UI Header
st.markdown("<h1>✨ 🌌 AI Vision Prompt Pro</h1>", unsafe_allow_html=True)
st.markdown("<h3>AI Image to Hyper-Detailed Prompt</h3>", unsafe_allow_html=True)
st.write("Target an image to generate ultra-detailed prompts for AI art generators.")

st.divider()

# 6. Layout: 2 Column Design
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("#### 📷 Input Image Source")
    uploaded_file = st.file_uploader("Upload Image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Original Input Preview', use_container_width=True)

with col2:
    st.markdown("#### 🤖 AI Generation Results")
    if uploaded_file:
        if st.button('🚀 GENERATE ULTRA-DETAILED PROMPT'):
            with st.spinner('Analyzing intricate details...'):
                # Processing
                processor, model, device = load_model()
                inputs = processor(image, return_tensors="pt").to(device)
                out = model.generate(**inputs, max_new_tokens=100)
                base_caption = processor.decode(out[0], skip_special_tokens=True)

                # --- HYPER DETAILED LOGIC (The "Magic" Prompt) ---
                ultra_detailed_prompt = (
                    f"**Ultra-detailed professional photography of {base_caption}.** "
                    "Every surface is rendered with incredibly fine tactile textures. "
                    "Cinematic volumetric atmospheric lighting, high-end 85mm prime lens, "
                    "with very soft bokeh. Shot at ISO 100 with global illumination, "
                    "ray tracing, and rule of thirds. Professional digital art, "
                    "Unreal Engine 5 render style with subsurface scattering on all elements."
                )

                # Results Display
                st.markdown("##### ## Masterpiece Prompt")
                st.markdown(f'<div class="prompt-box">"{ultra_detailed_prompt}"</div>', unsafe_allow_html=True)
                
                st.markdown("##### 🔥 Negative Prompt")
                st.warning("blurry, low quality, distorted, grainy, low resolution, bad anatomy, deformed limbs, out of frame")
                
                st.success("✅ Ready to copy!")
    else:
        st.info("Waiting for image upload to begin analysis...")

st.markdown("---")
st.caption("Developed by SRI ANANYA | AI Image to Prompt Generation System")
