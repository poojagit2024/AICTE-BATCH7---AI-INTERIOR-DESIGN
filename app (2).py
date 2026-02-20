import os
import streamlit as st
import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
import requests
import base64
from io import BytesIO

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Interior Design Generator", layout="wide")

STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")  # set via env variable
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    llm = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base"
    ).to(device)

    return blip_processor, blip_model, tokenizer, llm

blip_processor, blip_model, tokenizer, llm = load_models()

# ---------------- FUNCTIONS ----------------
def analyze_room(image):
    inputs = blip_processor(image, return_tensors="pt").to(device)
    output = blip_model.generate(**inputs, max_length=50)
    return blip_processor.decode(output[0], skip_special_tokens=True)

def generate_design(description, style):
    prompt = f"""
    Room description: {description}
    Interior style: {style}

    Suggest:
    - Furniture
    - Color palette
    - Lighting
    - Decor ideas
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm.generate(**inputs, max_length=250)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_image(prompt):
    if not STABILITY_API_KEY:
        st.warning("Stability API key not set. Image generation skipped.")
        return None

    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": 7,
        "height": 1024,
        "width": 1024,
        "samples": 1,
        "steps": 30
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        st.error(f"Stability API Error {response.status_code}")
        try:
            st.json(response.json())
        except:
            st.write(response.text)
        return None

    data = response.json()

    if "artifacts" not in data:
        st.error("No image returned by Stability AI")
        st.json(data)
        return None

    img_base64 = data["artifacts"][0]["base64"]
    return Image.open(BytesIO(base64.b64decode(img_base64)))

# ---------------- UI ----------------
st.title("🏠 AI-Powered Interior Design Generator")

uploaded = st.file_uploader(
    "Upload a room image",
    type=["jpg", "jpeg", "png"]
)

style = st.selectbox(
    "Select Interior Style",
    ["Minimal", "Scandinavian", "Modern", "Luxury"]
)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Room", use_column_width=True)

    with st.spinner("Analyzing room..."):
        description = analyze_room(image)

    st.subheader("🧠 Room Analysis")
    st.write(description)

    with st.spinner("Generating design suggestions..."):
        ideas = generate_design(description, style)

    st.subheader("🎨 Interior Design Suggestions")
    st.write(ideas)

    if st.button("Generate AI Redesigned Room (Optional)"):
        with st.spinner("Generating image..."):
            img_prompt = f"{style} interior design, {description}"
            result_img = generate_image(img_prompt)

            if result_img:
                st.image(result_img, caption="AI Generated Design")
