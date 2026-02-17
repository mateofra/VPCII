import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from transformers import ViTModel, ViTImageProcessor, SwinModel, AutoImageProcessor, CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pickle

# 1. Configuración
DEVICE = "cpu" # No teu caso usamos CPU
DATASET_PATH = "./IMagenet/tiny-imagenet-200/train/"
SAVE_PATH = "./precomputed_data/"
BATCH_SIZE = 64 # Podes subilo a 128 se ves que sobra RAM

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# 2. Cargar Modelos e Procesadores
print("Cargando modelos...")
models = {
    "vit": (ViTModel.from_pretrained('google/vit-base-patch16-224'), 
            ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')),
    "swin": (SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224"), 
             AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")),
    "dino": (ViTModel.from_pretrained("facebook/dino-vitb16"), 
             ViTImageProcessor.from_pretrained("facebook/dino-vitb16")),
    "clip": (CLIPModel.from_pretrained("openai/clip-vit-base-patch32"), 
             CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"))
}

for name in models:
    models[name][0].to(DEVICE).eval()

# 3. Preparar Dataset
# Usamos unha transformación xenérica para o DataLoader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

# Gardamos as rutas das imaxes para identificalas logo
image_paths = [s[0] for s in dataset.samples]
with open(os.path.join(SAVE_PATH, "image_paths.pkl"), "wb") as f:
    pickle.dump(image_paths, f)

# 4. Extracción
features = {
    "vit": [], "swin": [], "dino": [], "clip": []
}

print(f"Iniciando extracción de 100,000 imaxes en {DEVICE}...")

with torch.no_grad():
    for imgs, _ in tqdm(loader):
        imgs = imgs.to(DEVICE)
        
        # ViT
        feat_vit = models["vit"][0](imgs).last_hidden_state[:, 0, :]
        features["vit"].append(F.normalize(feat_vit, p=2, dim=1).cpu())
        
        # Swin
        feat_swin = models["swin"][0](imgs).pooler_output
        features["swin"].append(F.normalize(feat_swin, p=2, dim=1).cpu())
        
        # DINO
        feat_dino = models["dino"][0](imgs).last_hidden_state[:, 0, :]
        features["dino"].append(F.normalize(feat_dino, p=2, dim=1).cpu())
        
        # CLIP (Visual)
        feat_clip = models["clip"][0].get_image_features(pixel_values=imgs)
        features["clip"].append(F.normalize(feat_clip, p=2, dim=1).cpu())

# 5. Gardar resultados
print("Gardando tensores...")
for name in features:
    tensor_final = torch.cat(features[name])
    torch.save(tensor_final, os.path.join(SAVE_PATH, f"features_{name}.pt"))

print("¡Proceso finalizado! Xa podes abrir a App.")