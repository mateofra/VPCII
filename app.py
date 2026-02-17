import streamlit as st
import torch
import torch.nn.functional as F
from transformers import ViTModel, ViTImageProcessor, SwinModel, AutoImageProcessor, CLIPModel, CLIPProcessor
from PIL import Image
import pickle
import os

st.set_page_config(page_title="Buscador PRO", layout="wide")

# Función para cargar os datos precalculados
@st.cache_resource
def load_precomputed_data():
    base_path = "./precomputed_data/"
    data = {
        "vit": torch.load(base_path + "features_vit.pt"),
        "swin": torch.load(base_path + "features_swin.pt"),
        "dino": torch.load(base_path + "features_dino.pt"),
        "clip": torch.load(base_path + "features_clip.pt"),
    }
    with open(base_path + "image_paths.pkl", "rb") as f:
        paths = pickle.load(f)
    return data, paths

# Carga de modelos só para procesar a QUERY (unha soa imaxe/texto)
@st.cache_resource
def load_query_models():
    # Cargamos o necesario para procesar a entrada do usuario
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # ... cargar os outros igual que no script anterior ...
    return clip_model, clip_proc

# --- INTERFACE ---
st.title("⚡ Buscador Ultra-Rápido (100k imaxes)")

try:
    all_features, all_paths = load_precomputed_data()
    clip_model, clip_proc = load_query_models()
    
    col1, col2 = st.columns([1, 3])

    with col1:
        st.header("Entrada")
        mode = st.radio("Modo", ["Imaxe", "Texto", "Combinado"])
        model_name = st.selectbox("Modelo", ["vit", "swin", "dino", "clip"])
        
        query_img = st.file_uploader("Imaxe", type=['jpg', 'png'])
        query_text = st.text_input("Texto", "a photo of a cat")
        top_k = st.slider("Resultados", 5, 20, 10)

    if st.button("Executar Busca"):
        # 1. Extraer vector da query (esto é rápido porque é só UNHA imaxe)
        # Aquí usarías a lóxica de extracción do script anterior...
        # Exemplo para Texto con CLIP:
        with torch.no_grad():
            inputs = clip_proc(text=[query_text], return_tensors="pt", padding=True)
            q_feat = clip_model.get_text_features(**inputs)
            q_feat = F.normalize(q_feat, p=2, dim=1)

        # 2. BUSCA MATRICIAL (O "MAXIA")
        # Comparar 1x768 contra 100.000x768
        target_features = all_features[model_name]
        sims = torch.mm(q_feat, target_features.t()) # ¡Isto leva milisegundos!
        
        values, indices = torch.topk(sims, top_k)

        # 3. Mostrar
        st.header("Resultados")
        cols = st.columns(5)
        for i, idx in enumerate(indices[0]):
            with cols[i % 5]:
                img_path = all_paths[idx]
                st.image(img_path, caption=f"Score: {values[0][i]:.3f}")

except FileNotFoundError:
    st.error("Primeiro debes executar 'python precompute.py' para xerar os datos.")