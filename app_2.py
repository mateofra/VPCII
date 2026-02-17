import streamlit as st
import torch
import torch.nn.functional as F
from transformers import ViTModel, ViTImageProcessor, SwinModel, AutoImageProcessor, CLIPModel, CLIPProcessor
from PIL import Image
import pickle
import os

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Buscador VPC II", layout="wide")
DEVICE = "cpu"

# --- FUNCIÓNS DE CARGA ---
@st.cache_resource
def load_precomputed_data():
    base_path = "./datos_precomputados/"
    try:
        data = {
            "vit": torch.load(base_path + "features_vit.pt", map_location=DEVICE),
            "swin": torch.load(base_path + "features_swin.pt", map_location=DEVICE),
            "dino": torch.load(base_path + "features_dino.pt", map_location=DEVICE),
            "clip": torch.load(base_path + "features_clip.pt", map_location=DEVICE),
        }
        with open(base_path + "image_paths.pkl", "rb") as f:
            paths = pickle.load(f)
        return data, paths
    except:
        return None, None

@st.cache_resource
def load_single_model(model_name):
    """Carga só o modelo que imos usar (máis rápido e aforra RAM)"""
    if model_name == "vit":
        return ViTModel.from_pretrained('google/vit-base-patch16-224').eval(), ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    elif model_name == "swin":
        return SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224").eval(), AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    elif model_name == "dino":
        return ViTModel.from_pretrained("facebook/dino-vitb16").eval(), ViTImageProcessor.from_pretrained("facebook/dino-vitb16")
    elif model_name == "clip":
        return CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval(), CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- EXTRACCIÓN ---
def get_query_feat(img, model_name):
    model, proc = load_single_model(model_name)
    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        if model_name == "clip":
            out = model.get_image_features(**inputs)
            return F.normalize(getattr(out, "image_embeds", getattr(out, "pooler_output", out)), p=2, dim=1)
        elif model_name == "swin":
            return F.normalize(model(**inputs).pooler_output, p=2, dim=1)
        else:
            return F.normalize(model(**inputs).last_hidden_state[:, 0, :], p=2, dim=1)

def get_text_feat(text):
    model, proc = load_single_model("clip")
    inputs = proc(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        return F.normalize(getattr(out, "text_embeds", getattr(out, "pooler_output", out)), p=2, dim=1)

def fix_path(path):
    idx = path.find("tiny-imagenet-200")
    return os.path.join(".", path[idx:]) if idx != -1 else path

# --- INTERFACE (Amosase de inmediato) ---
all_features, all_paths = load_precomputed_data()

if all_features:
    col1, col2 = st.columns([1, 3])

    with col1:
        st.header("Entrada")
        mode = st.radio("Modo", ["Imaxe", "Texto (CLIP)", "Combinado"])
        
        # Selección de modelo
        if mode == "Imaxe":
            model_choice = st.selectbox("Modelo", ["vit", "swin", "dino", "clip"])
        else:
            st.info("Usando CLIP para texto")
            model_choice = "clip"
            
        # Widgets de entrada (Sempre visibles)
        uploaded_file = st.file_uploader("Subir imaxe", type=['jpg', 'png', 'jpeg'])
        query_text = st.text_input("Descrición", "a photo of a cat")
        
        if mode == "Combinado":
            alpha = st.slider("Peso Imaxe vs Texto", 0.0, 1.0, 0.5)
            
        top_k = st.slider("Resultados", 5, 25, 10)
        btn_buscar = st.button("🚀 Buscar")

    # --- LÓXICA AO PREMER O BOTÓN ---
    if btn_buscar:
        q_feat = None
        with st.spinner("Cargando modelo e procesando..."):
            try:
                if mode == "Imaxe" and uploaded_file:
                    img = Image.open(uploaded_file).convert("RGB")
                    q_feat = get_query_feat(img, model_choice)
                
                elif mode == "Texto (CLIP)":
                    q_feat = get_text_feat(query_text)
                    
                elif mode == "Combinado" and uploaded_file:
                    img = Image.open(uploaded_file).convert("RGB")
                    f_img = get_query_feat(img, "clip")
                    f_txt = get_text_feat(query_text)
                    q_feat = F.normalize(alpha * f_img + (1-alpha) * f_txt, p=2, dim=1)

                if q_feat is not None:
                    # Busca
                    targets = all_features[model_choice]
                    sims = torch.mm(q_feat, targets.t())
                    scores, indices = torch.topk(sims, top_k)

                    # Mostrar en Col2
                    with col2:
                        st.header(f"Resultados ({model_choice.upper()})")

                        # 1. Cambiamos de 3 columnas a 2 (ou incluso 1 se as queres xigantes)
                        grid = st.columns(2) 

                        for i, idx in enumerate(indices[0]):
                            # 2. Axustamos o índice para que reparta en 2 columnas
                            with grid[i % 2]: 
                                path = fix_path(all_paths[idx])
                                
                                # 3. Usamos use_container_width=True para que a imaxe 
                                # se estire ata ocupar todo o ancho da columna
                                st.image(Image.open(path), 
                                caption=f"Score: {scores[0][i]:.3f}", 
                                use_container_width=True)
                else:
                    st.error("Falta a imaxe ou o texto")
            except Exception as e:
                st.error(f"Erro: {e}")
else:
    st.error("Non se atoparon os datos precomputados.")

