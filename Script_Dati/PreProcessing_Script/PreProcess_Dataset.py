import os
import cv2
import numpy as np
import shutil  # <--- NUOVA IMPORTAZIONE
from tqdm import tqdm
from PreProcessing import MacenkoNormalizer, HistologyPreprocessor

DATASET_DIR = "E:\\MarcoCariccia\\Dataset\\Dataset_Tiles"

def process_dataset(reference_image_path):
    # 1. Inizializzazione (con i parametri alpha e beta ottimizzati)
    normalizer = MacenkoNormalizer(alpha=5, beta=0.05)
    
    ref_img = cv2.imread(reference_image_path)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    normalizer.fit(ref_img)
    
    preprocessor = HistologyPreprocessor(color_normalizer=normalizer)

    wsi_folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
    
    for wsi in tqdm(wsi_folders, desc="Processing WSIs"):
        images_src = os.path.join(DATASET_DIR, wsi, "images")
        images_dst = os.path.join(DATASET_DIR, wsi, "images_preprocessed")
        
        if not os.path.exists(images_src):
            continue
            
        # --- LOGICA DI PULIZIA AUTOMATICA ---
        # Se la cartella esiste già, la elimina con tutto il suo contenuto
        if os.path.exists(images_dst):
            shutil.rmtree(images_dst)
        
        # Crea la cartella da zero
        os.makedirs(images_dst)
        # ------------------------------------
        
        for filename in os.listdir(images_src):
            if filename.endswith(".png"):
                img_path = os.path.join(images_src, filename)
                save_path = os.path.join(images_dst, filename)
                
                img = cv2.imread(img_path)
                if img is None: continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                processed_img = preprocessor(img_rgb)
                
                processed_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, processed_bgr)

if __name__ == "__main__":
    # IMPORTANTE: Scegli un tile rappresentativo (buona colorazione, tessuto visibile) come riferimento
    REF_PATH = "E:\MarcoCariccia\Dataset\img_ref.jpg"
    process_dataset(REF_PATH)