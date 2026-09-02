"""
config.py
====================================================================
UNICO punto in cui si modificano percorsi e iperparametri.
Le funzioni parametri_*() sono l'unica fonte di verita' per i moduli
che non devono dipendere da config (perdite.py) e per il report.
====================================================================
"""

import os
import torch

# ====================================================================
# AUGMENTATION DI COLORAZIONE  —  variabile dell'ablation
# ====================================================================
USA_HED = True
P_HED = 0.80

K_INTENSITA    = 3.0     # amplificazione massima delle concentrazioni H&E
K_ATTENUAZIONE = 1.6     # attenuazione massima (fattore minimo = 1/K_ATT) 1.6 se K_INTENSITA >= 2.0, 1.4 se K_INTENSITA < 2.0
K_RAPPORTO     = 1.3     # sbilanciamento fra ematossilina ed eosina
BIAS_REL       = 0.02
SCALA_H        = 1.02
SCALA_E        = 0.25

def parametri_hed():
    return {"k_intensita": K_INTENSITA, "k_attenuazione": K_ATTENUAZIONE,
            "k_rapporto": K_RAPPORTO, "bias_relativo": BIAS_REL,
            "scala_h": SCALA_H, "scala_e": SCALA_E, "p": P_HED}

# ====================================================================
# PERCORSI
# ====================================================================
DATASET_DIR = r"E:\Tirocinio\Dataset\Dataset_Tiles_768"
PATH_TRAIN_CSV = r"E:\Tirocinio\Progetto\UNet_768\Dataset_UNet\train_split_vincolato.csv"
PATH_TEST_CSV = r"E:\Tirocinio\Progetto\UNet_768\Dataset_UNet\test_split_vincolato.csv"
CARTELLA_IMG = "images_preprocessed"
CARTELLA_MSK = "masks"

BASE_RISULTATI = r"E:\Tirocinio\Progetto\UNet_768\Risultati"
NOME_RUN = f"K{K_INTENSITA}_A{K_ATTENUAZIONE}_R{K_RAPPORTO}"
RISULTATI_DIR = os.path.join(BASE_RISULTATI, NOME_RUN)

DIR_MODELLI  = os.path.join(RISULTATI_DIR, "modelli")
DIR_METRICHE = os.path.join(RISULTATI_DIR, "metriche")
DIR_GRAFICI  = os.path.join(RISULTATI_DIR, "grafici")
DIR_CACHE    = os.path.join(BASE_RISULTATI, "cache")

LIVELLO_NORMALIZZAZIONE = "L1"

PREFISSO_CKPT = "privato_fold"
NOME_STORIA = "storia_privato.json"
NOME_REPORT = "report_privato.json"
NOME_DIAGNOSI = "diagnosi_dataset.json"
NOME_PLOT = "curve_privato.png"

ESTENSIONI = (".png",)

def crea_cartelle():
    for d in (RISULTATI_DIR, DIR_MODELLI, DIR_METRICHE, DIR_GRAFICI, DIR_CACHE):
        os.makedirs(d, exist_ok=True)

def path_modello(k):
    return os.path.join(DIR_MODELLI, f"{PREFISSO_CKPT}{k}.pth")

def path_metrica(nome):
    return os.path.join(DIR_METRICHE, nome)

def path_grafico(nome):
    return os.path.join(DIR_GRAFICI, nome)

# ====================================================================
# CLASSI E GERARCHIA
# ====================================================================
IGNORE_INDEX = 255          # non annotato, presunto TESSUTO
IGNORE_VETRO = 254          # non annotato, riconosciuto come VETRO
CLASSE_SFONDO = 0

NOMI_FINI = ["Sfondo", "CIN1", "Ghiandole", "HSIL", "Mucosa", "Stroma"]
N_CLASSI_FINI = 6

NOMI_LAST = ["GLASS", "NEGATIVE", "LSIL", "HSIL"]
GRUPPI_LAST_FINI = [[0], [2, 4, 5], [1], [3]]
LUT_LAST_FINI = [0, 2, 1, 3, 1, 1]
IDX_GLASS_LAST = 0
IDX_NEGATIVE_LAST = 1
IDX_LSIL_LAST = 2
IDX_HSIL_LAST = 3
NOMI_POSNEG = ["NEGATIVO", "POSITIVO"]

MODALITA_CLASSI = 6

if MODALITA_CLASSI == 6:
    NUM_CLASSES = N_CLASSI_FINI
    CLASS_NAMES = list(NOMI_FINI)
    GRUPPI_LAST = [list(g) for g in GRUPPI_LAST_FINI]
    LUT_LAST = torch.tensor(LUT_LAST_FINI, dtype=torch.long)
    RIMAPPA_MASCHERE = False
elif MODALITA_CLASSI == 4:
    NUM_CLASSES = 4
    CLASS_NAMES = list(NOMI_LAST)
    GRUPPI_LAST = [[0], [1], [2], [3]]
    LUT_LAST = torch.arange(4, dtype=torch.long)
    RIMAPPA_MASCHERE = True
else:
    raise ValueError("MODALITA_CLASSI deve essere 6 o 4")

CLASSI_TESSUTO = [c for c in range(NUM_CLASSES) if c != CLASSE_SFONDO]
CLASSI_VETRO = [c for c in range(NUM_CLASSES) if c not in CLASSI_TESSUTO]

# ====================================================================
# PSEUDO-ETICHETTA VETRO / TESSUTO
# ====================================================================
SOGLIA_OD_VETRO = 0.03      # densita' ottica media sotto cui il pixel e' vetro

def parametri_vetro():
    return {"soglia_od": SOGLIA_OD_VETRO,
            "ignore_index": IGNORE_INDEX,
            "valore_vetro": IGNORE_VETRO}

# ====================================================================
# MODELLO E OTTIMIZZAZIONE
# ====================================================================
ENCODER = "timm-mobilenetv3_large_100"
ENCODER_FALLBACK = ["mobilenet_v2", "timm-efficientnet-b0", "resnet18", "resnet34"]   
DROPOUT_HEAD = 0.15

STAGE_SCHEDULE = {0: 0, 5: 1}
FRAZIONE_SBLOCCO = 0.50

TILE = 512
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 30
WARMUP_EP = 2
LR_DECODER = 3e-4
LR_ENCODER = 5e-6
WD_DECODER = 1e-2
WD_ENCODER = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 12
SEED = 42

# ====================================================================
# REGOLARIZZAZIONE
# ====================================================================
# CutMix scartato: il taglio rettangolare recide l'asse basale-superficie
# dell'epitelio, che e' l'asse su cui e' definito il criterio LAST.
USA_CUTMIX = False
LABEL_SMOOTH = 0.10

# ====================================================================
# PESI LOSS E BILANCIAMENTO
# ====================================================================
W_CE      = 0.40
W_DICE    = 0.25
W_LAST    = 0.20
W_CLINICA = 0.15
W_TESSUTO = 0.02

SCE_ALPHA, SCE_BETA, SCE_A = 1.0, 0.3, 4.0
FTL_ALPHA, FTL_BETA, FTL_GAMMA, FTL_SMOOTH = 0.3, 0.7, 1.333, 1.0

# Omega[vera, predetta]. Ordine: Sfondo, CIN1, Ghiandole, HSIL, Mucosa, Stroma.
# CIN1 -> Mucosa costa 1.00, CIN1 -> HSIL costa 0.35: penalizzato quasi tre
# volte tanto, ma non infinite volte come nella penalita' binaria, che
# assegnando costo zero a CIN1 -> HSIL aveva un minimo degenere.
MATRICE_COSTI = [
    [0.00, 0.20, 0.20, 0.30, 0.20, 0.20],   # vera Sfondo
    [0.90, 0.00, 0.70, 0.35, 1.00, 0.70],   # vera CIN1
    [0.50, 0.30, 0.00, 0.40, 0.30, 0.30],   # vera Ghiandole
    [0.90, 0.50, 0.80, 0.00, 1.00, 0.80],   # vera HSIL
    [0.40, 0.25, 0.30, 0.50, 0.00, 0.30],   # vera Mucosa
    [0.60, 0.30, 0.30, 0.40, 0.30, 0.00],   # vera Stroma
]
if MODALITA_CLASSI == 4:
    W_LAST = 0.0
    MATRICE_COSTI = None

SENS_HSIL_MINIMA  = 0.90
RECALL_NEG_MINIMA = 0.75

N_FOLD = 5
FOLD_DA_ESEGUIRE = None

USA_SAMPLER = True
SAMPLING_ALPHA = 0.5
WSI_BALANCE = 0.7
CLIP_PESI = (0.5, 3.0)

SOGLIA_PRESENZA_CLASSE = 0.005
SOGLIA_WSI_MINIME = 2 * N_FOLD
SOGLIA_COOCCORRENZA = 3

def parametri_loss():
    """
    Unico punto di verita' per la loss: passato a LossGerarchica con **
    e serializzato tale e quale nel report, cosi' i parametri effettivi
    del run non possono divergere da quelli documentati.
    """
    return {
        "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX,
        "ignore_vetro": IGNORE_VETRO,
        "gruppi_last": [list(g) for g in GRUPPI_LAST],
        "lut_last": LUT_LAST.tolist(),
        "classi_tessuto": list(CLASSI_TESSUTO),
        "classi_vetro": list(CLASSI_VETRO),
        "w_ce": W_CE, "w_dice": W_DICE, "w_last": W_LAST,
        "w_clinica": W_CLINICA, "w_tessuto": W_TESSUTO,
        "sce_alpha": SCE_ALPHA, "sce_beta": SCE_BETA, "sce_a": SCE_A,
        "label_smooth": LABEL_SMOOTH,
        "ftl_alpha": FTL_ALPHA, "ftl_beta": FTL_BETA,
        "ftl_gamma": FTL_GAMMA, "ftl_smooth": FTL_SMOOTH,
        "matrice_costi": MATRICE_COSTI,
    }

def parametri_selezione():
    return {"sens_hsil_minima": SENS_HSIL_MINIMA,
            "recall_neg_minima": RECALL_NEG_MINIMA}

def firma_esperimento():
    return f"Tile: {TILE} | K_INT: {K_INTENSITA} | K_ATT: {K_ATTENUAZIONE} | Batch: {BATCH_SIZE}"