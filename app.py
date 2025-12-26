import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="AI Vision Prompt Pro", page_icon="🎨", layout="wide")

# 2. Optimized CSS for Visibility and Clean UI (Removing Extra Boxes)
st.markdown("""
    <style>
    /* Main Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #a855f7 0%, #ec4899 50%, #facc15 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1e1b4b;
    }

    /* Clean White Cards for Input/Output */
    .custom-card {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0px 15px 35px rgba(0,0,0,0.1);
        color: #1f2937;
        margin-bottom: 20px;
    }

    /* Fixing Visibility - Heading and Subtitle in White */
    .main-title {
        color: white !important;
        font-size: 45px !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 5px;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
    }
    .main-subtitle {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 18px !important;
        text-align: center;
        margin-bottom: 40px;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        font-weight: bold;
        border: none;
        box-shadow: 0px 8px 15px rgba(16, 185, 129, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Model Loading Logic
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model, device

# 4. Sidebar Content
with st.sidebar:
    st.markdown("<h2 style='color:white;'>👤 Project Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.info("**Model:** BLIP Vision AI")
    st.info("**Mode:** Ultra-Detailing")
    st.divider()
    st.caption("AI-Powered Prompt Generation System")

# 5. Fixed Main Header (Ensures words are visible)
st.markdown('<p class="main-title">✨ AI Image to Ultra-Detailed Prompt</p>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Generate hyper-realistic, masterpiece-level prompts for AI Art Generators.</p>', unsafe_allow_html=True)

# 6. Main Content - 2 Column Layout (Removing extra manual boxes)
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📷 Input Image Source")
    uploaded_file = st.file_uploader("Drop your image file here", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI Generation Results")
    if uploaded_file:
        if st.button('🚀 GENERATE ULTRA-DETAILED PROMPT'):
            with st.spinner('Performing deep analysis...'):
                processor, model, device = load_model()
                inputs = processor(image, return_tensors="pt").to(device)
                out = model.generate(**inputs, max_new_tokens=100)
                base_caption = processor.decode(out[0], skip_special_tokens=True)

                ultra_prompt = (
                    f"**Ultra-detailed professional photography of {base_caption}.** "
                    "Fine tactile textures, cinematic lighting, 85mm prime lens, f/1.8, "
                    "global illumination, ray tracing, masterpiece quality, photorealistic."
                )

                st.markdown("#### 🔥 Masterpiece Prompt")
                st.info(ultra_prompt)
                st.success("Analysis Complete!")
    else:
        st.write("Waiting for image upload to begin analysis...")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><center style='color: white;'>Developed by SRI ANANYA | Professional AI Vision System</center>", unsafe_allow_html=True)
