import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
import sys
import os
import onnxruntime as ort

# --- App config ---
st.set_page_config(page_title="Anomaly Detection UI", layout="wide")
st.title("Anomaly Detection Interface")

# --- Sidebar model selection ---
model_option = st.sidebar.selectbox("Choose a model:", ["INP-Former", "EfficientAD", "GLASS"], index=0)

# === Model-specific heatmap min/max (empirically computed) ===
MODEL_STATS = {
    "INP-Former": {"min": 0, "max": 0.5},
    "EfficientAD": {"min": -4.75, "max": -1.25},  # update if needed
    "GLASS": {"min": 0.15, "max": 1.0},        # update if needed
}
TOP_K_PERCENT = 0.007

# --- Sliders ---
user_pred_thresh = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, step=0.01)
user_vis_thresh = st.sidebar.slider("Heatmap Display Threshold", 0.0, 1.0, 0.3, step=0.01)

# --- Strip-based cropping ---
def crop_dark_edges(image_np, darkness_threshold=30, strip_width=5):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    left = 0
    right = w - 1
    while left + strip_width < right:
        if np.mean(gray[:, left:left+strip_width]) > darkness_threshold:
            break
        left += strip_width
    while right - strip_width > left:
        if np.mean(gray[:, right-strip_width:right]) > darkness_threshold:
            break
        right -= strip_width
    return image_np[:, left:right+1]

# === EfficientAD Setup ===
@st.cache_resource
def load_efficientad():
    sys.path.append('/content/drive/MyDrive/Neural_Networks_Project/EfficientAD')
    from common import get_pdn_small, get_autoencoder
    from efficientad import teacher_normalization, map_normalization, predict
    from torchvision import transforms
    from torch.serialization import add_safe_globals
    import torch.nn as nn

    add_safe_globals({"Sequential": nn.Sequential})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_dir = "/content/drive/MyDrive/Neural_Networks_Project/EfficientAD/output/1/trainings/mvtec_ad/wood"

    teacher = torch.load(f"{weights_dir}/teacher_final.pth", map_location=device, weights_only=False)
    student = torch.load(f"{weights_dir}/student_final.pth", map_location=device, weights_only=False)
    autoencoder = torch.load(f"{weights_dir}/autoencoder_final.pth", map_location=device, weights_only=False)

    teacher.to(device).eval()
    student.to(device).eval()
    autoencoder.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return teacher, student, autoencoder, 0.5, 0.1, 0.05, 0.95, 0.05, 0.95, device, transform, predict

# === Model Implementations ===
def generate_glass_heatmap(image):
    onnx_model_path = "/content/drive/MyDrive/Neural_Networks_Project/glass_simplified.onnx"
    input_size = (256, 256)
    expected_batch_size = 8

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_pil = Image.fromarray(image)
    img_tensor = transform(image_pil).unsqueeze(0).numpy()

    if img_tensor.shape[0] != expected_batch_size:
        padded = np.zeros((expected_batch_size, *img_tensor.shape[1:]), dtype=img_tensor.dtype)
        padded[0] = img_tensor[0]
        img_tensor = padded

    ort_session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    outputs = ort_session.run(None, {"input": img_tensor})
    return outputs[0][0]

def generate_efficientad_heatmap(image):
    teacher, student, autoencoder, mean, std, q1, q2, q3, q4, device, transform, predict = load_efficientad()

    image_pil = Image.fromarray(image)
    img_tensor = transform(image_pil).unsqueeze(0).to(device)

    map_combined, _, _ = predict(img_tensor, teacher, student, autoencoder,
                                 mean, std, q1, q2, q3, q4)

    return 1.0 - map_combined.squeeze().cpu().numpy()

def generate_inpformer_heatmap(image):
    input_size = 392
    onnx_model_path = "/content/drive/MyDrive/Neural_Networks_Project/INP-Former/inpformer.onnx"
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    resized = cv2.resize(image, (input_size, input_size))
    normed = (resized / 255.0 - mean) / std
    transposed = np.transpose(normed, (2, 0, 1))
    input_tensor = np.expand_dims(transposed, axis=0).astype(np.float32)

    session = ort.InferenceSession(onnx_model_path)
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    enc_feats, dec_feats = outputs[:2], outputs[2:4]

    maps = []
    for fs, ft in zip(enc_feats, dec_feats):
        fs, ft = fs[0], ft[0]
        fs, ft = np.transpose(fs, (1, 2, 0)), np.transpose(ft, (1, 2, 0))
        sim = 1 - np.sum(fs * ft, axis=2) / (np.linalg.norm(fs, axis=2) * np.linalg.norm(ft, axis=2) + 1e-8)
        sim = cv2.resize(sim, (256, 256))
        maps.append(sim)

    return cv2.GaussianBlur(np.mean(maps, axis=0), (5, 5), sigmaX=4)

# === UI Logic ===
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    if "heatmap" not in st.session_state or st.session_state.get("last_model") != model_option or st.session_state.get("last_file") != uploaded_file.name:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        img_array = crop_dark_edges(img_array)
        image = Image.fromarray(img_array).resize((256, 256))
        img_array = np.array(image)
        st.session_state["img_array"] = img_array

        if model_option == "GLASS":
            heatmap = generate_glass_heatmap(img_array)
        elif model_option == "EfficientAD":
            heatmap = generate_efficientad_heatmap(img_array)
        elif model_option == "INP-Former":
            heatmap = generate_inpformer_heatmap(img_array)
        else:
            st.stop()

        st.session_state["heatmap"] = heatmap
        st.session_state["last_model"] = model_option
        st.session_state["last_file"] = uploaded_file.name
    else:
        img_array = st.session_state["img_array"]
        heatmap = st.session_state["heatmap"]

    col1, col2 = st.columns(2)
    col1.subheader("Input Image")
    col1.image(img_array, use_container_width=True)

    flat = heatmap.flatten()
    k = max(1, int(len(flat) * TOP_K_PERCENT))
    topk_mean = np.mean(np.partition(flat, -k)[-k:])
    model_min = MODEL_STATS[model_option]["min"]
    model_max = MODEL_STATS[model_option]["max"]
    real_thresh = model_min + (model_max - model_min) * user_pred_thresh
    pred_label = "Defect" if topk_mean > real_thresh else "Good"
    normalized_score = (topk_mean - model_min) / (model_max - model_min + 1e-8)
    normalized_score = np.clip(normalized_score, 0, 1)

    vis_thresh = model_min + (model_max - model_min) * user_vis_thresh
    heatmap_draw = np.where(heatmap >= vis_thresh, heatmap, 0)
    heatmap_vis = (heatmap_draw - model_min) / (model_max - model_min + 1e-8)
    heatmap_vis = np.clip(heatmap_vis, 0, 1)

    heatmap_color = cv2.applyColorMap((heatmap_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_color = cv2.resize(heatmap_color, (img_array.shape[1], img_array.shape[0]))
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_color, 0.4, 0)

    col2.subheader(f"Prediction: {pred_label}")
    col2.markdown(f"Anomaly Score: `{normalized_score:.2f}`")
    col2.image(overlay, use_container_width=True)
