"""
config.py
====================================================================
UNICO punto in cui si modificano percorsi e iperparametri.

Nessun altro modulo contiene costanti hard-coded: se un valore va
cambiato, si cambia qui.

Struttura attesa del dataset:

    DATASET_DIR/
        Prova 1/
            images/                  <- tile originali
            images_preprocessed/     <- tile normalizzati (usati)
            masks/                   <- maschere PNG indicizzate
        Prova 2/
            ...
====================================================================
"""

import os
import torch

# ====================================================================
# PERCORSI
# ====================================================================
DATASET_DIR = r"E:\Tirocinio\Dataset\Dataset_Tiles"
CARTELLA_IMG = "images_preprocessed"
CARTELLA_MSK = "masks"

RISULTATI_DIR = r"E:\Tirocinio\Progetto\UNet\Risultati_3.0"
DIR_MODELLI = os.path.join(RISULTATI_DIR, "modelli")
DIR_METRICHE = os.path.join(RISULTATI_DIR, "metriche")
DIR_GRAFICI = os.path.join(RISULTATI_DIR, "grafici")
DIR_CACHE = os.path.join(RISULTATI_DIR, "cache")

# Il nome della cartella non dice quale livello di normalizzazione
# contiene. Va dichiarato QUI e finisce nel report, perche' in tesi
# serve poter scrivere quale preprocessing e' stato usato.
#   "L1" = solo allineamento della tinta verso la matrice H&E canonica
#   "L2" = L1 + riscalatura delle concentrazioni verso la mediana
#   "?"  = da verificare nel report di PreProcess_Dataset.py
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
# CLASSI
# ====================================================================
# ATTENZIONE: qui 0 e' una CLASSE VERA (vetro), l'ignore e' 255
# ("Tissue": tessuto delimitato ma non classificato). E' l'opposto di
# MTCHI, dove 0 = fuori RoI = ignore. Non invertire.
IGNORE_INDEX = 255
CLASSE_SFONDO = 0

NOMI_FINI = ["Sfondo", "CIN1", "Ghiandole", "HSIL", "Mucosa", "Stroma"]
N_CLASSI_FINI = 6

# --------------------------------------------------------------------
# GERARCHIA LAST / WHO 2020
# --------------------------------------------------------------------
NOMI_LAST = ["GLASS", "NEGATIVE", "LSIL", "HSIL"]
GRUPPI_LAST_FINI = [[0], [2, 4, 5], [1], [3]]
#                  vetro  ghiand+mucosa+stroma  CIN1  HSIL
LUT_LAST_FINI = [0, 2, 1, 3, 1, 1]
IDX_GLASS_LAST = 0
IDX_LSIL_LAST = 2
IDX_HSIL_LAST = 3

# ====================================================================
# ABLATION: granularita' della supervisione
#   6 = il modello predice le 6 classi fini, piu' un termine di loss
#       ausiliario sulle probabilita' marginalizzate LAST.
#       La compattezza intra-classe fine e' PRESERVATA: Stroma e Mucosa
#       restano uscite distinte, spinte via dai termini a 6 classi.
#       Il termine LAST cambia solo QUANTO COSTA un errore, non le
#       etichette.
#   4 = le maschere vengono rimappate su LAST al caricamento. Il modello
#       non vede mai le classi fini, e NEGATIVE deve davvero assorbire
#       ghiandole, mucosa e stroma insieme. E' la condizione di
#       controllo per l'obiezione sulla varianza intra-classe.
# ====================================================================
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

# Classi che non sono vetro: servono al vincolo sui pixel 255
CLASSI_TESSUTO = [c for c in range(NUM_CLASSES) if c != CLASSE_SFONDO]

# ====================================================================
# MODELLO
# ====================================================================
ENCODER = "timm-mobilenetv3_large_100"
ENCODER_FALLBACK = ["mobilenet_v2", "timm-efficientnet-b0", "resnet18", "resnet34"]
DECODER_CHANNELS = (96, 64, 48, 32, 16)
DROPOUT_HEAD = 0.15

# Stage 2 (encoder completo) non esiste: su MTCHI lo sblocco totale
# coincideva esattamente con il minimo di val_loss, dopo il quale la
# validazione divergeva.
STAGE_SCHEDULE = {0: 0, 5: 1}
FRAZIONE_SBLOCCO = 0.35        # calcolata sui PARAMETRI, non sui tensori

# ====================================================================
# OTTIMIZZAZIONE
# ====================================================================
TILE = 512                     
BATCH_SIZE = 12
NUM_WORKERS = 4
EPOCHS = 30
WARMUP_EP = 2           # numero di epoche di warmup (solo decoder, LR crescente)
LR_DECODER = 3e-4       # learning rate per il decoder (molto piu' alto)
LR_ENCODER = 1e-5       # learning rate per l'encoder (molto piu' basso)
WD_DECODER = 1e-2       # weight decay per il decoder
WD_ENCODER = 1e-4       # weight decay per l'encoder
GRAD_CLIP = 1.0        # clip dei gradienti
PATIENCE = 8        # numero di epoche senza miglioramento prima di ridurre LR
SEED = 42

# ====================================================================
# REGOLARIZZAZIONE
# ====================================================================
USA_CUTMIX = True
P_CUTMIX = 0.5
CUTMIX_BETA = 1.0
LABEL_SMOOTH = 0.05

USA_HED = True
P_HED = 0.80

# --- NUOVI PARAMETRI HED MOLTIPLICATIVO (Ablation) ---
# K_INTENSITA = 1.4 (In-domain), 2.0 (Intermedia), 3.0 (Cross-domain MTCHI)
K_INTENSITA = 3.0  
K_RAPPORTO  = 1.3
BIAS_REL    = 0.02
SCALA_H     = 1.02
SCALA_E     = 0.25

# ====================================================================
# PESI DEI TERMINI DI LOSS
# ====================================================================
W_CE = 0.40                    # cross-entropy sulle classi fini
W_DICE = 0.30                  # Dice sulle classi fini
W_LAST = 0.25                  # CE sulle probabilita' marginalizzate LAST
W_TESSUTO = 0.05               # vincolo "i pixel 255 non sono vetro"

if MODALITA_CLASSI == 4:
    W_LAST = 0.0               # ridondante: le etichette sono gia' LAST

# ====================================================================
# VALIDAZIONE E BILANCIAMENTO
# ====================================================================
N_FOLD = 5
FOLD_DA_ESEGUIRE = None        # None = tutti; [0] per provarne uno solo

USA_SAMPLER = True
SAMPLING_ALPHA = 0.5
WSI_BALANCE = 0.7              # esponente del cap per WSI
CLIP_PESI = (0.3, 5.0)

# Frazione di pixel oltre la quale si dice che una WSI "contiene" una classe
SOGLIA_PRESENZA_CLASSE = 0.005

# Sotto questo numero di WSI, una classe non e' considerata validabile:
# resta addestrata, ma viene esclusa dalla METRICA DI SELEZIONE e
# segnalata nel report. Con Stroma in ~10 WSI e l'83% dei pixel in 2
# sole WSI, la media fra fold non descriverebbe nulla di reale.
SOGLIA_WSI_MINIME = 2 * N_FOLD

# Sotto questo numero di WSI di co-occorrenza, il confine fra due classi
# non e' validabile: sono quasi separate per vetrino, quindi il modello
# puo' distinguerle riconoscendo il vetrino invece della morfologia
# (Howard et al. 2021).
SOGLIA_COOCCORRENZA = 3