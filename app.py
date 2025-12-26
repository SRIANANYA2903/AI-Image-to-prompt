import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="AI Vision Prompt Pro", page_icon="🎨", layout="wide")

# 2. Modern UI Design CSS (Matching your reference image)
st.markdown("""
    <style>
    /* Main Background Gradient */
    .main {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 50%, #ff7eb3 100%);
        color: white;
    }
    
    /* Modern Card Styling */
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.2);
        color: #333;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1e1e2f;
    }
    .sidebar-text { color: white; font-weight: 600; }

    /* Titles */
    h1, h2, h3 { color: white !important; text-align: center; font-weight: 800; }
    .card-title { color: #1e1e2f !important; font-weight: 700; font-size: 22px; margin-bottom: 15px; }

    /* Button Design */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0px 5px 15px rgba(0,0,0,0.3); }

    /* Prompt Box */
    .ultra-prompt-box {
        background-color: #fdfdfd;
        border-left: 5px solid #ff7eb3;
        padding: 15px;
        border-radius: 10px;
        font-size: 15px;
        color: #444;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Model (CPU/GPU Auto-detect)
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model, device

# 4. Sidebar - Project Dashboard
with st.sidebar:
    st.markdown("<h2 style='color:white;'>👤 Project Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.info("**Model:** BLIP-AI")
    st.info("**Category:** Image-to-Prompt")
    st.markdown("---")
    st.success("New Update: Ultra-Detail Mode Active")

# 5. Header Section
st.markdown("<h1>✨ AI Image to Ultra-Detailed Prompt</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Generate hyper-realistic, masterpiece-level prompts for AI Art Generators.</p>", unsafe_allow_html=True)
st.divider()

# 6. Main Content - 2 Column Layout
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<p class="card-title">📷 Input Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True, caption="Source Image Ready")

with col2:
    st.markdown('<p class="card-title">🤖 AI Generation Results</p>', unsafe_allow_html=True)
    if uploaded_file:
        if st.button('🚀 GENERATE ULTRA-DETAILED PROMPT'):
            with st.spinner('AI analyzing details...'):
                processor, model, device = load_model()
                inputs = processor(image, return_tensors="pt").to(device)
                out = model.generate(**inputs, max_new_tokens=100)
                base_caption = processor.decode(out[0], skip_special_tokens=True)

                # Ultra Detailed Logic
                masterpiece = (
                    f"**Ultra-detailed professional photography of {base_caption}.** "
                    "Every surface is rendered with incredibly fine tactile textures. "
                    "Cinematic lighting, 85mm lens, f/1.8, global illumination, ray tracing, "
                    "masterpiece, Unreal Engine 5 render, 8k resolution, photorealistic."
                )

                st.markdown("##### 🔥 Ultra-Prompt")
                st.markdown(f'<div class="ultra-prompt-box">{masterpiece}</div>', unsafe_allow_html=True)
                st.success("✅ Analysis Complete!")
    else:
        st.markdown("<p style='color:#666;'>Upload an image to see AI results here.</p>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Developed by SRI ANANYA | Professional AI Vision System")
