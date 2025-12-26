import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="AI Vision Prompt Pro", page_icon="🎨", layout="wide")

# 2. Modern UI CSS (Reference Design Matching)
st.markdown("""
    <style>
    /* Gradient Background covering the whole page */
    .stApp {
        background: linear-gradient(135deg, #6a11cb 0%, #ff7eb3 50%, #2575fc 100%);
        background-attachment: fixed;
    }
    
    /* White Glassmorphism Cards */
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        background-color: rgba(255, 255, 255, 0.92);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0px 15px 35px rgba(0,0,0,0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 20px;
    }

    /* Professional Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e1e2f;
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Titles and Typography */
    h1, h2, h3 { 
        color: white !important; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    .card-label { 
        color: #1e1e2f !important; 
        font-weight: 700; 
        font-size: 20px; 
        margin-bottom: 10px;
        display: block;
    }

    /* Generate Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.8em;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        font-weight: bold;
        border: none;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0px 6px 20px rgba(0,0,0,0.3);
    }

    /* Output Box Styling */
    .output-box {
        background-color: #ffffff;
        border-left: 6px solid #ff7eb3;
        padding: 20px;
        border-radius: 12px;
        font-size: 16px;
        color: #333;
        line-height: 1.6;
        box-shadow: inset 0px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Model Loading Logic (Auto-Detect CPU/GPU)
@st.cache_resource
def load_model():
    # Detects if GPU is available to prevent crashes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model, device

# 4. Sidebar Content
with st.sidebar:
    st.markdown("<h2 style='color:white; text-align:left;'>👤 Project Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='color:#bbb;'>System Stats</p>", unsafe_allow_html=True)
    st.info("**Model:** BLIP Vision v1.0")
    st.info("**Mode:** Ultra-Detailing")
    st.divider()
    st.caption("AI-Powered Prompt Engineering System")

# 5. Header Section
st.markdown("<h1>✨ AI Image to Ultra-Detailed Prompt</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>Transform any image into a professional-grade AI art prompt.</p>", unsafe_allow_html=True)
st.divider()

# 6. Main Content - 2 Column Layout
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<span class="card-label">📷 Input Image</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop your image file here", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)

with col2:
    st.markdown('<span class="card-label">🤖 AI Generation Results</span>', unsafe_allow_html=True)
    if uploaded_file:
        if st.button('🚀 GENERATE ULTRA-DETAILED PROMPT'):
            with st.spinner('AI is performing deep analysis...'):
                processor, model, device = load_model()
                inputs = processor(image, return_tensors="pt").to(device)
                out = model.generate(**inputs, max_new_tokens=100)
                base_caption = processor.decode(out[0], skip_special_tokens=True)

                # --- HYPER DETAILED PROMPT ENGINEERING ---
                masterpiece = (
                    f"**Ultra-detailed professional photography of {base_caption}.** "
                    "Every surface is rendered with incredibly fine tactile textures. "
                    "Cinematic lighting, 85mm prime lens, f/1.8, ISO 100, global illumination, "
                    "ray tracing, masterpiece quality, Unreal Engine 5 render style, "
                    "8k resolution, highly intricate, photorealistic."
                )

                st.markdown("##### 🔥 Masterpiece Prompt")
                st.markdown(f'<div class="output-box">{masterpiece}</div>', unsafe_allow_html=True)
                st.success("Analysis Complete!")
    else:
        st.info("Waiting for image upload to begin analysis...")
