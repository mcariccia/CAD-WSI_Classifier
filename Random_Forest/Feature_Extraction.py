import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from skimage import color, filters
from skimage.color import separate_stains, hed_from_rgb
from skimage.feature import structure_tensor, hessian_matrix_det


DATASET_DIR = "/home/marco-cariccia/University/Terzo_Anno/Tirocinio/Dataset/Dataset_Tiles"
SPLITS = ["Dataset_RF/train_split_RF.csv", "Dataset_RF/test_split_RF.csv"]

COLOR_MAPPING = {
    (255, 192, 203): 1,   # CIN1 (Esempio: Rosso)
    (0, 255, 0): 2,   # Endocervical_glands (Esempio: Verde)
    (0, 0, 255): 3,   # HSIL (Esempio: Blu)
    (255, 255, 0): 4, # Normal_Mucosa (Esempio: Giallo)
    (0, 255, 255): 5, # Stroma (Esempio: Ciano)
    (0, 0, 0): 0      # Sfondo/Ignore
}

MAX_IMG_FOR_WSI = 100
MAX_PIXEL_FOR_IMG = 100   




#=================================================================================================================

def extract_multiscale_features(channel, sigmas=[1, 5, 15]):
    feats = []
    for s in sigmas:
        gauss = filters.gaussian(channel, sigma=s)
        lap = filters.laplace(gauss)

        Axx, Axy, Ayy = structure_tensor(gauss, sigma=s)
        trace = Axx + Ayy
        tmp = np.sqrt((Axx - Ayy)**2 + 4*Axy**2)
        lambda1 = (trace + tmp) / 2
        lambda2 = (trace - tmp) / 2
        coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-8)

        hdet = hessian_matrix_det(gauss, sigma=s)

        feats.append(np.stack([gauss, lap, coherence, hdet], axis=-1))

    return np.concatenate(feats, axis=-1)

#=====================================================================================================================


def build_gabor_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 4):
        for sigma in (3, 5):
            lamda = np.pi / 2.0
            gamma = 0.5
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters

GABOR_FILTERS = build_gabor_filters()

#================================================================================================================

def calculate_local_std(img_channel, kernel_size=(5,5)):
    img_float = np.float32(img_channel)
    mean_sq = cv2.blur(img_float ** 2, kernel_size)
    sq_mean = cv2.blur(img_float, kernel_size) ** 2
    var = cv2.max(mean_sq - sq_mean, 0)
    return cv2.sqrt(var)

#===============================================================================

def extract_super_features(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: return None
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Feature HED 
    hed = separate_stains(img_rgb, hed_from_rgb)
    hematoxylin = hed[:, :, 0]
    dab = hed[:, :, 2]
    
    # 2. Feature Colore
    R, G, B = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    all_channels = [R, G, B, hematoxylin, dab]
    feature_maps = []
    
    # --- Calcolo Multiscala (60 feature) ---
    for ch in all_channels:
        feature_maps.append(extract_multiscale_features(ch, sigmas=[1, 5, 15]))

    # --- Calcolo Gabor (8 feature) ---
    for kern in GABOR_FILTERS:
        fimg = cv2.filter2D(gray, cv2.CV_32F, kern)
        feature_maps.append(fimg[..., np.newaxis])

    feats = np.concatenate(feature_maps, axis=-1)
    
    feats = np.nan_to_num(feats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return feats

#======================================================================================

def get_label_mask(mask_path, mapping):
    mask_bgr = cv2.imread(mask_path)
    h, w = mask_bgr.shape[:2]
    
    label_mask = np.zeros((h, w), dtype=np.uint8)
    
    for rgb_color, class_id in mapping.items():
        bgr_color = rgb_color[::-1] 
        match = np.all(mask_bgr == bgr_color, axis=-1)
        label_mask[match] = class_id
        
    return label_mask[..., np.newaxis]

#=======================================================================================

def select_set_imgs(split):
    all_data_pairs = []

    wsi_list = []
    with open(split, 'r') as f:
        for row in f:
            wsi_list.append(row.strip())

    for wsi in wsi_list:
        img_dir = os.path.join(DATASET_DIR, str(wsi), "images")
        mask_dir = os.path.join(DATASET_DIR, str(wsi), "masks")

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            continue

        possible_pairs = []
        for filename in os.listdir(img_dir):
            if filename.endswith(".png"):
                target_mask_path = os.path.join(mask_dir, filename)
                if os.path.exists(target_mask_path):
                    img_path = os.path.join(img_dir, filename) 
                    possible_pairs.append((img_path, target_mask_path))

        n_to_pick = min(len(possible_pairs), MAX_IMG_FOR_WSI)
        selected_samples = random.sample(possible_pairs, n_to_pick)   

        all_data_pairs.extend(selected_samples)                              

    return(all_data_pairs)


#====================================
#FLUSSO DI ESECUZIONE
#====================================

if __name__ == "__main__":
    for split in SPLITS:
        first_write = True
        set_pairs = select_set_imgs(split)
        output_csv = split.replace(".csv", "_dataset.csv")

        print(f"\n--- Processamento del file: {split} ---")

        for img, mask in tqdm(set_pairs, desc="Estrazione Pixel"):
            features = extract_super_features(img)
            
            if features is None:
                continue
                
            label = get_label_mask(mask, COLOR_MAPPING)

            gray_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

            valid_mask = (label[..., 0] != 0) & (gray_img < 220)
            valid_y, valid_x = np.where(valid_mask)

            if len(valid_y) > 0:
                n_to_pick = min(len(valid_y), MAX_PIXEL_FOR_IMG)

                chosen_indices = np.random.choice(len(valid_y), n_to_pick, replace=False)

                selected_y = valid_y[chosen_indices]
                selected_x = valid_x[chosen_indices]

                features_subset = features[selected_y, selected_x] 
                labels_subset = label[selected_y, selected_x, 0]   

                num_features = features_subset.shape[1]
                
                feature_names = [f'feat_{j}' for j in range(num_features)]
                
                df = pd.DataFrame(features_subset, columns=feature_names)
                
                df['target'] = labels_subset
                df.to_csv(output_csv, mode='a', index=False, header=first_write)

                first_write = False