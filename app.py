import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="AI Vision Prompt Pro", page_icon="🎨", layout="wide")

# 2. Custom CSS for Professional Look
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .prompt-box {
        padding: 25px;
        border-radius: 15px;
        background-color: #ffffff;
        color: #000000;
        border-left: 10px solid #2e7d32;
        font-size: 16px;
        font-weight: 500;
        line-height: 1.6;
        margin-bottom: 20px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Model Function (GPU-to-CPU Auto-switch)
@st.cache_resource
def load_model():
    # Detect if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model, device

# 4. Main Header
st.title("✨ AI Image to Ultra-Detailed Prompt")
st.write("Generate hyper-realistic, masterpiece-level prompts for AI Art Generators.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Input Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Original Input', use_container_width=True)

with col2:
    st.subheader("🤖 AI Generation Results")
    if uploaded_file:
        if st.button('🚀 GENERATE ULTRA-DETAILED PROMPT'):
            with st.spinner('Analyzing every pixel for deep details...'):
                processor, model, device = load_model()
                inputs = processor(image, return_tensors="pt").to(device)
                out = model.generate(**inputs, max_new_tokens=100)
                base_caption = processor.decode(out[0], skip_special_tokens=True)

                # --- HYPER-DETAILED PROMPT LOGIC ---
                # Ingat dhaan neenga keta maari details add aagudhu
                ultra_prompt = (
                    f"**Ultra-detailed professional photography of {base_caption}.** "
                    "Every surface is rendered with incredibly fine tactile textures. "
                    "Cinematic volumetric atmospheric lighting, high-end 85mm prime lens, "
                    "with sharp focus on subject. Shot at ISO 100 with global illumination, "
                    "ray tracing, and subsurface scattering. Unreal Engine 5 render style, "
                    "photorealistic, 8k resolution, highly intricate details, masterpiece quality."
                )

                st.markdown("### 🔥 Masterpiece Prompt")
                st.markdown(f'<div class="prompt-box">{ultra_prompt}</div>', unsafe_allow_html=True)
                
                st.markdown("### 🚫 Negative Prompt (To avoid bad results)")
                st.warning("blurry, low quality, distorted, grainy, low resolution, bad anatomy, out of frame, pixelated, abstract")
                
                st.success("✅ Ultra-Detailed Analysis Complete!")
    else:
        st.info("Waiting for image upload to begin analysis...")
