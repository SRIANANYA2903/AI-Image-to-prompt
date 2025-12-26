import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="AI Vision Prompt Pro", page_icon="🎨", layout="wide")

# 2. Custom CSS for High Visibility & Professional Look
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .prompt-box {
        padding: 25px;
        border-radius: 15px;
        background-color: #ffffff; 
        color: #000000;            
        border-left: 10px solid #2e7d32;
        font-size: 18px;
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
        font-size: 16px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar with Info
with st.sidebar:
    st.title("👨‍💻 Project Dashboard")
    st.markdown("---")
    st.info("**Model:** BLIP (Vision-Language)")
    st.info("**Category:** Image-to-Prompt")
    st.divider()
    st.write("Target: High-Detail Image Replication")

# 4. Main Header
st.title("✨ AI Image to Hyper-Realistic Prompt")
st.write("Upload an image and let AI generate a masterpiece-level prompt.")

# 5. Load Model Function (Cached) - Corrected Indentation
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model, device

# 6. Main Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Input Image Source")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Original Input', use_container_width=True)

with col2:
    st.subheader("🤖 AI Generation Results")
    if uploaded_file:
        if st.button('🚀 GENERATE REALISTIC PROMPT'):
            with st.spinner('Analyzing details...'):
                # Model Inference
                processor, model, device = load_model()
                inputs = processor(image, return_tensors="pt").to(device)
                out = model.generate(**inputs, max_new_tokens=65)
                base_caption = processor.decode(out[0], skip_special_tokens=True)
                
                # Masterpiece Prompt
                masterpiece_prompt = (
                    f"**Hyper-realistic professional photography of** {base_caption}. "
                    "Cinematic lighting, 8k resolution, photorealistic masterpiece."
                )

                st.markdown("### 🔥 Masterpiece Prompt")
                st.markdown(f'<div class="prompt-box">{masterpiece_prompt}</div>', unsafe_allow_html=True)
                st.success("✅ Complete!")
    else:
        st.info("Waiting for image upload...")
