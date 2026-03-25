import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor, CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import os

# 1. Configuración de dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Buscador", layout="wide")

# Funcion que toma los datos precomputados
# guardados en ./precomputed_data
@st.cache_resource
def load_data():
    base = "./precomputed_data/"
    modelos = ["vit", "swin", "dino", "clip"]
    features = {}
    
    for modelo in modelos:
        archivo = f"{base}features_{modelo}.pt" # construimos un str del archivo
        if os.path.exists(archivo):
            features[modelo] = torch.load(archivo, map_location=device)
    
    with open(base + "image_paths.pkl", "rb") as f:
        rutas_pkl = pickle.load(f) # Guardamos el archivo image_paths.pkl
    return features, rutas_pkl

# Función llamada cuando se realiza una busquedad
# devuelve un modelo y su procesador
# carga un solo modelo cada vez para reducir el tiempo
@st.cache_resource
def get_model(name):
    if name == "clip":
        m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    elif name == "dino":
        m = AutoModel.from_pretrained("facebook/dino-vitb16").to(device).eval()
        p = AutoImageProcessor.from_pretrained("facebook/dino-vitb16")
    elif name == "vit":
        m = AutoModel.from_pretrained("google/vit-base-patch16-224").to(device).eval()
        p = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    elif name == "swin":
        m = AutoModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224").to(device).eval()
        p = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    return m, p
    
@st.cache_resource
def get_category_mapping():
    mapping = {}
    # Ruta ao ficheiro words.txt 
    path_words = "./IMagenet/tiny-imagenet-200/words.txt"
    if os.path.exists(path_words):
        with open(path_words, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    # parts[0] é o ID, parts[1] é o nome 
                    mapping[parts[0]] = parts[1].split(",")[0] 
    return mapping

# Función que devuelve el mapa de calor
def get_similarity_map(model, processor, query_img_tensor, target_img_path):
    target_img = Image.open(target_img_path).convert("RGB")
    t_input = processor(images=target_img, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Extraer embeddings
        q_out = model(query_img_tensor.to(device))
        q_cls = F.normalize(q_out.last_hidden_state[:, 0, :], p=2, dim=-1)
        
        # Extraer parches do resultado 
        t_out = model(**t_input)
        t_patches = t_out.last_hidden_state[:, 1:, :] 
        t_patches = F.normalize(t_patches, p=2, dim=-1)
        
        # Calcular similitude
        sim = torch.matmul(q_cls, t_patches.transpose(1, 2)).squeeze() 
        
        # Normalizar para visualización
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        
        # DINO v1 imaxes 224 -> 14x14
        sim_grid = sim.view(14, 14).cpu().numpy()
        
        return cv2.resize(sim_grid, target_img.size, interpolation=cv2.INTER_CUBIC)

#######################################################
# Interfaz
######################################################
st.title("Buscador de Imaxes - Proxecto I VPCII")
st.image("./baner.png")
try:
    all_feats, rutas = load_data()
    # cargamos los archivos .pt
    
    with st.sidebar:
        # declaramos os elementos que aparecen na páxina
        st.header("Busqueda avanzada")
        seleccionador_modelo = st.pills("Selecciona un modelo", ["clip", "vit", "swin", "dino"])
        modo = st.pills("Modo de búsqueda", ["Imaxe", "Texto"])
        n_resultados = st.slider("Nº resultados", 2, 40, 18)
        n_col = st.slider("Nº columnas", 2, 20, 6)
        if modo == "Imaxe":
            uploaded_file = st.file_uploader("Subir imaxe ↴", type=['jpg', 'jpeg', 'png'])
        else:
            query_text = st.text_input("Escribe unha descripción da imaxe:", "a small image of a cat")
        st.checkbox("I agree with the terms of service")


    if st.button("Iniciar búsqueda"):
        
        
        # utilizamos a función get_model para obter o modelo e processor
        # do modelo seleccionado
        if seleccionador_modelo != None:
            model, processor = get_model(seleccionador_modelo)
        else:
            st.error("Selecciona un modelo para buscar.")
            st.stop()
        
        with torch.no_grad():
            if modo == "Texto":
                if seleccionador_modelo != "clip": # comprobamos que o modelo sea clip e devolvemos error se non
                    st.error("A busca por texto require o uso do modelo CLIP.")
                    st.stop()
                
                inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(device)
                outputs = model.get_text_features(**inputs)
                
                q_embeding = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
                if q_embeding.ndim == 1:
                    q_embeding = q_embeding.unsqueeze(0)
                
                q_embeding = F.normalize(q_embeding, p=2, dim=-1)

            else: # Modo Imaxe
                if uploaded_file is None: # devolvemos error se non hay imaxe
                    st.warning("Por favor, sube unha imaxe para realizar a búsqueda.")
                    st.stop()
                
                imaxe_subida = Image.open(uploaded_file).convert("RGB")
                inputs = processor(images=imaxe_subida, return_tensors="pt").to(device)
                
                if seleccionador_modelo == "clip":
                    outputs = model.get_image_features(**inputs)
 
                    q_embeding = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
                    if q_embeding.ndim == 1:
                        q_embeding = q_embeding.unsqueeze(0)
                else:
                    q_output = model(**inputs)
                    q_embeding = q_output.last_hidden_state[:, 0, :]
                
                q_embeding = F.normalize(q_embeding, p=2, dim=-1)

        # 2. Busca 
        target_features = all_feats[seleccionador_modelo].to(device)
        
        if q_embeding.ndim == 1:
            q_embeding = q_embeding.unsqueeze(0)

        # 
        sims = torch.mm(q_embeding, target_features.t())
        sims_display = (sims + 1) / 2 
        values, indices = torch.topk(sims_display, n_resultados) #seleccionamos as n mellores imaxes
        
        category_names = get_category_mapping()
        # 3. Mostrar Resultados
        st.header(f"Top {n_resultados} Resultados")
        n_cols = 2
        for i in range(0, len(indices[0]), n_col):
            cols = st.columns(n_col)
            for j in range(n_col):
                idx_in_batch = i + j
                if idx_in_batch < len(indices[0]):
                    idx = indices[0][idx_in_batch].item()
                    with cols[j]:
                        ruta_img = rutas[idx]
                        score = values[0][idx_in_batch]
                        
                        folder_id = ruta_img.split('/')[-3] 
                        cat_name = category_names.get(folder_id, "Descoñecida")
                        
                        st.markdown(f"#### Resultado {idx_in_batch + 1}")
                        st.write(f"**Categoría:** :orange[{cat_name.upper()}]") # En cor e maiúsculas
                        st.write(f"**Confianza:** {score:.2%}")
                        
                        orig_img = Image.open(ruta_img).convert("RGB")
                        
                        if modo == "Imaxe" and seleccionador_modelo == "dino":
                            heatmap = get_similarity_map(model, processor, inputs['pixel_values'], ruta_img)
                            fig, ax = plt.subplots(figsize=(10, 10))
                            ax.imshow(orig_img)
                            ax.imshow(heatmap, cmap='magma_r', alpha=0.5, extent=(0, orig_img.size[0], orig_img.size[1], 0))
                            ax.axis('off')
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.image(orig_img, use_container_width=True)

except Exception as e:
    st.info("Sube unha imaxe ou escribe un texto e preme Buscar.")
    st.write(f"Erro: {e}")
