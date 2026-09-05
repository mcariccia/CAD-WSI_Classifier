"""
config_esterno.py
====================================================================
Percorsi e macro dell'estrazione tile da MTCHI Task 2.
Solo costanti: niente logica.
====================================================================
"""

import os

# ====================================================================
# PERCORSI
# ====================================================================
# Dataset MTCHI originale (struttura ufficiale <partizione>/Img, /Truth)
DIR_MTCHI = r"E:\DatasetMTCHI\Dataset\Task2"
PARTIZIONI = ["Training", "Test"]

DIR_USCITA = r"E:\Tirocinio\Dataset\Dataset_Tiles_MTCHI"

CARTELLA_IMG = "images"
CARTELLA_MSK = "masks"
NOME_REPORT = "report_estrazione.json"

# Nome delle cartelle delle singole RoI, tutte allo stesso livello sotto
# DIR_USCITA (le partizioni Training/Test NON diventano cartelle: la
# provenienza resta nel report, campo 'partizione').
#   "originale"   -> Crop_B1204944-3-7D   contiene il paziente nel nome
#   "progressivo" -> WSI_MTCHI_01, 02, ...
SCHEMA_CARTELLE = "originale"


# ====================================================================
# GRIGLIA
# ====================================================================
TILE = 512
OVERLAP = 0.0

ALLINEA_BORDI = True

EVITA_DOPPIO_CONTEGGIO = True

# ====================================================================
# NERO FUORI RoI
# ====================================================================
SOGLIA_NERO = 0             # nero pieno: tutti e tre i canali <= questo valore

AREA_MINIMA_NERO = 5000     # px; sotto questa soglia non e' fondo, e' tessuto
DILATA_NERO = 2

SOSTITUISCI_NERO = True
VALORE_BIANCO = 255


# ====================================================================
# QUALI TILE TENERE
# ====================================================================
MIN_FRAZIONE_ANNOTAZIONE = 0.0


# ====================================================================
# ETICHETTE MTCHI
# ====================================================================
VALORI_AMMESSI = {0, 1, 2, 3, 4}
NOMI_MTCHI = {0: "Background", 1: "Normal", 2: "CIN1", 3: "CIN2", 4: "CIN3"}

ESTENSIONI = (".png", ".jpg", ".tif", ".tiff")


def nome_cartella(nome_file, indice):
    """indice e' 1-based e serve solo allo schema 'progressivo'."""
    if SCHEMA_CARTELLE == "progressivo":
        return f"WSI_MTCHI_{indice:02d}"
    return os.path.splitext(nome_file)[0]


def cartella_roi(nome_cartella_):
    return os.path.join(DIR_USCITA, nome_cartella_)


# ####################################################################
#                              T E S T
# ####################################################################

DIR_PIPELINE = r"E:\Tirocinio\Progetto\UNet"

# Checkpoint dei 5 fold, una cartella per valore di K.
DIR_MODELLI = {
    "k2": r"E:\Tirocinio\Progetto\UNet\Risultati\K2.0_A1.6_R1.3\modelli",
    "k3": r"E:\Tirocinio\Progetto\UNet\Risultati\K3.0_A1.6_R1.3\modelli",
}
PREFISSO_CKPT = "privato_fold"
N_FOLD = 5

DIR_RISULTATI = r"E:\Tirocinio\Progetto\Test_Esterno\Risultati_Esterno"
CARTELLA_PREPROCESSED = "images_preprocessed"

# --------------------------------------------------------------------
# LE QUATTRO CONFIGURAZIONI
# --------------------------------------------------------------------
# Disegno fattoriale 2x2 su due variabili indipendenti. Senza tutte e quattro
# le celle i due effetti restano confusi: un K piu' ampio potrebbe sembrare
# utile solo perche' supplisce a una normalizzazione mancante, o viceversa.
#
# K=2.0 e' la configurazione PRIMARIA pre-specificata: vince sull'endpoint
# clinico interno e non usa alcuna informazione su MTCHI.
# K=3.0 e' SECONDARIA: il suo intervallo e' stato calibrato sulle statistiche
# di densita' ottica di MTCHI (hed_augment.py), quindi la sua valutazione qui
# NON e' cieca rispetto al dominio target e va etichettata come tale.
CONFIGURAZIONI = [
    {"nome": "vahadane_no_k2", "vahadane": False, "k": "k2", "primaria": True},
    {"nome": "vahadane_si_k2", "vahadane": True,  "k": "k2", "primaria": False},
    {"nome": "vahadane_no_k3", "vahadane": False, "k": "k3", "primaria": False},
    {"nome": "vahadane_si_k3", "vahadane": True,  "k": "k3", "primaria": False},
]

CONFIG_ESEMPI = "vahadane_si_k2"

# --------------------------------------------------------------------
# SPAZIO DELLE CLASSI
# --------------------------------------------------------------------
IGNORE = 255
NUM_CLASSI = 3
NOMI_CLASSI = ["NEGATIVE", "LSIL", "HSIL"]
IDX_NEGATIVE, IDX_LSIL, IDX_HSIL = 0, 1, 2

NUM_CLASSI_PRIVATO = 6
NOMI_PRIVATO = ["Sfondo", "CIN1", "Ghiandole", "HSIL", "Mucosa", "Stroma"]

# ground truth MTCHI -> spazio di confronto.
# 0 (Background) resta IGNORE: conflaggia nero fuori RoI, vetro, stroma e
# ghiandole non annotati. 
LUT_TRUTH = [IGNORE, IDX_NEGATIVE, IDX_LSIL, IDX_HSIL, IDX_HSIL]

# predizione a 6 classi -> spazio di confronto.
SEGNAPOSTO_VETRO = 254
LUT_PRED = [SEGNAPOSTO_VETRO,  # 0 Sfondo -> vetro
            IDX_LSIL,          # 1 CIN1
            IDX_NEGATIVE,      # 2 Ghiandole
            IDX_HSIL,          # 3 HSIL
            IDX_NEGATIVE,      # 4 Mucosa
            IDX_NEGATIVE]      # 5 Stroma

POLITICA_VETRO = "negativo"
LUT_INV_MTCHI = [1, 2, 1, 3, 1, 1]

# --------------------------------------------------------------------
# INFERENZA
# --------------------------------------------------------------------
BATCH_TEST = 16
# Su Windows i worker usano spawn e reimportano il modulo: funziona, ma se
# dovesse dare errori di pickle o accumulo di memoria, mettere 0.
NUM_WORKERS = 4
USA_TTA = True             # flip x4
USA_AMP = True

DECISIONI = ["argmax", "costo_minimo"]
DECISIONE_PRIMARIA = "argmax"

# --------------------------------------------------------------------
# STATISTICA
# --------------------------------------------------------------------
N_BOOTSTRAP = 2000
SEED = 42
N_MINIMO_PER_IC = 10

# Un tile entra nella statistica di rilevazione della classe c solo se almeno
# questa frazione dei suoi pixel appartiene a c. Sotto, il tile non contiene
# abbastanza lesione perche' "rilevata o no" significhi qualcosa.
FRAZIONE_MINIMA_LESIONE = 0.02
SOGLIE_RILEVAZIONE = (0.10, 0.25, 0.50)

SOGLIA_STRATIFICAZIONE = 0.20

# --------------------------------------------------------------------
# USCITE
# --------------------------------------------------------------------
ESPORTA_DAT = True          # ricompone le predizioni in spazio MTCHI, cosi'
                            # si puo' eseguire eval_task2.py ufficiale

# --------------------------------------------------------------------
# ESEMPI QUALITATIVI
# --------------------------------------------------------------------
SALVA_ESEMPI = True

# Frazione minima di pixel della classe target per considerare il tile un esempio valido. 
FRAZIONE_MINIMA_ESEMPIO = 0.05

# Percentili di IoU: caso difficile, tipico, riuscito. Mai il massimo.
PERCENTILI_ESEMPIO = ((10, "difficile"), (50, "tipico"), (90, "riuscito"))

MAX_ESEMPI_PER_ROI = 1

# Esempi di ERRORE: le confusioni piu' frequenti nella matrice aggregata.
N_ERRORI = 3
FRAZIONE_MINIMA_ERRORE = 0.15

N_CONFRONTI_DECISIONE = 2

DPI_ESEMPI = 200            # il privato usa 160; qui piu' alto per la stampa
FIGSIZE_ESEMPI = (18.0, 5.2)
ALPHA_OVERLAY = 0.55

COLORI_CLASSI = [(70, 175, 95),    # NEGATIVE  verde
                 (255, 160, 30),   # LSIL      arancione
                 (200, 35, 50)]    # HSIL      rosso
COLORE_NON_ANNOTATO = (70, 70, 70)
COLORE_VETRO_PRED = (245, 245, 245)   # bianco sporco, come Sfondo nel privato

COLORE_CORRETTO = (60, 170, 100)
COLORE_ERRORE = (225, 25, 90)


def dir_config(nome):
    return os.path.join(DIR_RISULTATI, nome)


def firma():
    return (f"MTCHI Task2 | {'+'.join(PARTIZIONI)} | tile {TILE} overlap "
            f"{OVERLAP} | TTA {'on' if USA_TTA else 'off'} | vetro "
            f"'{POLITICA_VETRO}' | min_annot {MIN_FRAZIONE_ANNOTAZIONE}")