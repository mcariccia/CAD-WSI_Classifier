"""
PreProcess_Dataset.py
====================================================================
Implementazione accademica basata su Vahadane (2016) e Tellez (2019).
Include:
- NMF Multi-restart robusta
- I0 (Bianco) dinamico calcolato per singola WSI
- Target allineato alla Mediana Normalizzata del dataset
- GESTIONE AUTOMATICA DELLE ANOMALIE IN VOLO (Niente script esterni)
====================================================================
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import NMF
import shutil
import json

# ====================================================================
# PARAMETRI
# ====================================================================
DATASET_DIR = r"E:\Tirocinio\Dataset\Dataset_Tiles_768"
CARTELLA_INPUT = "images"
CARTELLA_OUTPUT = "images_preprocessed"
NOME_REPORT = "report_normalizzazione.json"
ESTENSIONE = ".png"

ALPHA_SPARSITA = 0.1
MAX_PIXEL_FIT = 100000
PIXEL_PER_TILE_CAMPIONE = 3000    # pixel di TESSUTO campionati per tile
N_PIXEL_I0_PER_TILE = 2000        # pixel GREZZI per tile, per stimare il bianco I0
N_TILE_STIMA = 50                 # tile usati per la stima, scelti uniformemente
SOGLIA_TESSUTO_GREZZA = 220       # soglia rapida tessuto/vetro nel campionamento
SOGLIA_ALLARME_COSENO = 0.90

# La NMF va lasciata iterare: sotto le 1000 iterazioni la stima dei vettori
# degrada in modo brusco (verificato su matrici note: 16 gradi di errore medio
# a 500 iterazioni contro 3.6 a 1000). Il ConvergenceWarning di sklearn
# riguarda la convergenza delle CONCENTRAZIONI su centinaia di migliaia di
# pixel, non dei 6 numeri della matrice, che si assestano molto prima.
MAX_ITER_NMF = 1000
TOLLERANZA_ERRORE_NMF = 0.005     # soluzioni entro lo 0.5% = equivalenti
SEED = 42

PRIOR_HE = np.array([[0.5626, 0.2159],
                     [0.7201, 0.8012],
                     [0.4062, 0.5581]])
IO_DEFAULT = 256.0
# ====================================================================

class Vahadane:
    def __init__(self, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.restarts = [('nndsvda', seed), ('nndsvdar', seed), ('random', seed), ('random', seed + 1)]

    def get_I0(self, img_rgb):
        """
        Bianco di riferimento: 99mo percentile per canale.
        DEVE essere calcolato su pixel NON filtrati (vetro incluso): stimarlo
        dopo aver rimosso i pixel chiari sottostima I0 di ~20-30 livelli,
        scurendo sistematicamente l'output e introducendo una dominante.
        """
        if img_rgb.shape[0] == 0:
            return np.full(3, IO_DEFAULT)
        I0 = np.percentile(img_rgb, 99, axis=0)
        return np.maximum(I0, 1.0)

    def get_OD(self, img_rgb, I0):
        """
        Optical Density in logaritmo naturale.

        Si usa (px + 1) e non max(px, 1e-6): un pixel nero saturo (valore 0)
        con la seconda formula darebbe OD = -log(1e-6/244) = 19.3, un outlier
        3.5 volte piu' grande di qualunque densita' reale. La NMF minimizza
        l'errore quadratico, quindi pochi pixel di questo tipo (nuclei molto
        scuri, pigmento, artefatti) spostano i vettori di stain in modo
        sproporzionato. Con (px + 1) il massimo resta 5.5, sulla scala dei
        valori reali.
        """
        img_safe = img_rgb.astype(np.float64) + 1.0
        return -np.log(img_safe / I0)

    def get_tissue_mask(self, OD):
        return np.mean(OD, axis=-1) > 0.15

    def _normalizza_e_ordina(self, W):
        """Normalizza i vettori colore e assegna colonna 0 = H, colonna 1 = E."""
        norme = np.linalg.norm(W, axis=1, keepdims=True)
        if np.any(norme < 1e-8):
            return None
        HE = (W / norme).T
        sim = HE.T @ PRIOR_HE
        if sim[0, 0] + sim[1, 1] < sim[0, 1] + sim[1, 0]:
            HE = HE[:, [1, 0]]
        return HE

    def fit_wsi_matrix(self, pool_tessuto, pool_grezzo):
        """
        Stima i vettori di stain sul pool di TESSUTO e il bianco I0 sul pool
        GREZZO (vetro incluso). I due pool hanno scopi diversi e non vanno
        confusi: vedi la docstring di raccogli_pixel_wsi.
        """
        if pool_grezzo.shape[0] == 0:
            return None, np.full(3, IO_DEFAULT), "Nessun pixel campionato"

        I0_wsi = self.get_I0(pool_grezzo)          # <-- bianco dal pool NON filtrato

        if pool_tessuto.shape[0] == 0:
            return None, I0_wsi, "Nessun pixel di tessuto"

        OD = self.get_OD(pool_tessuto, I0_wsi)
        mask = self.get_tissue_mask(OD)
        OD_tissue = OD[mask]

        if OD_tissue.shape[0] < 1000:
            return None, I0_wsi, "Tessuto insufficiente"

        if OD_tissue.shape[0] > MAX_PIXEL_FIT:
            idx = self.rng.choice(OD_tissue.shape[0], MAX_PIXEL_FIT, replace=False)
            OD_tissue = OD_tissue[idx]

        # NMF Multi-restart
        soluzioni = []
        for init, rs in self.restarts:
            try:
                nmf = NMF(n_components=2, init=init, solver='cd', alpha_W=ALPHA_SPARSITA,
                          alpha_H=0.0, l1_ratio=1.0, max_iter=MAX_ITER_NMF,
                          tol=1e-4, random_state=rs)
                nmf.fit(OD_tissue)
                soluzioni.append((float(nmf.reconstruction_err_), nmf.components_))
            except Exception:
                continue

        if not soluzioni:
            return None, I0_wsi, "Fattorizzazione NMF fallita"

        # SELEZIONE: fra le soluzioni con errore di ricostruzione quasi
        # equivalente (entro TOLLERANZA_ERRORE_NMF), si sceglie quella piu'
        # compatibile con il prior H&E. Il solo errore minimo NON basta: la
        # NMF non e' identificabile in modo unico e a parita' di errore puo'
        # restituire coni piu' larghi del necessario. Verificato su dati
        # sintetici a matrice nota: solo-errore-minimo ~31 gradi di errore
        # medio, con questo tie-break ~3.6 gradi.
        errore_min = min(s[0] for s in soluzioni)
        migliore, punteggio_max = None, -np.inf
        for errore, W in soluzioni:
            if errore > errore_min * (1.0 + TOLLERANZA_ERRORE_NMF):
                continue
            HE_cand = self._normalizza_e_ordina(W)
            if HE_cand is None:
                continue
            punteggio = float(np.sum(HE_cand * PRIOR_HE))
            if punteggio > punteggio_max:
                punteggio_max, migliore = punteggio, HE_cand

        if migliore is None:
            migliore = self._normalizza_e_ordina(min(soluzioni, key=lambda s: s[0])[1])
        if migliore is None:
            return None, I0_wsi, "Vettori di stain degeneri"

        return migliore, I0_wsi, "OK"

    def transform_tile(self, img_rgb, HE_src, I0_wsi, HE_target):
        h, w, c = img_rgb.shape
        px = img_rgb.reshape(-1, 3)
        OD = self.get_OD(px, I0_wsi)
        
        # Deconvoluzione
        C = np.maximum(np.linalg.lstsq(HE_src, OD.T, rcond=None)[0], 0.0)
        
        # Ricostruzione
        # Inversa esatta di get_OD: se OD = -log((px+1)/I0) allora
        # px = I0*exp(-OD) - 1. Il "-1" deve esserci, altrimenti andata e
        # ritorno non coincidono e si introduce uno scostamento sistematico.
        Inorm = I0_wsi[:, None] * np.exp(-HE_target.dot(C)) - 1.0
        img_norm_flat = np.clip(Inorm.T, 0, 255).astype(np.uint8)
        
        # Risolto il bug delle dimensioni: maschera 1D applicata su array 1D
        mask_tissue = self.get_tissue_mask(OD)
        mask_bg = ~mask_tissue
        img_norm_flat[mask_bg] = px[mask_bg]
        
        return img_norm_flat.reshape(h, w, c)

def raccogli_pixel_wsi(images_src, files, normalizer):
    """
    Campiona pixel sparsi evitando sovraccarichi di memoria.

    Restituisce DUE pool distinti:
      - pool_tessuto : pixel di solo tessuto, per stimare i vettori di stain
      - pool_grezzo  : pixel NON filtrati, per stimare I0 (il bianco reale)

    I due pool devono restare separati: stimare I0 sul pool gia' filtrato
    significherebbe cercare il bianco dopo aver rimosso tutto cio' che e'
    bianco, sottostimandolo di ~20-30 livelli e introducendo sia uno
    scurimento sistematico sia una dominante di colore.

    I tile vengono scelti UNIFORMEMENTE lungo la slide: i nomi codificano le
    coordinate, quindi prendere i primi N in ordine alfabetico campionerebbe
    un solo angolo del vetrino.
    """
    pool_tessuto = []
    pool_grezzo = []
    n_per_tile = PIXEL_PER_TILE_CAMPIONE

    if not files:
        return np.empty((0, 3), dtype=np.uint8), np.empty((0, 3), dtype=np.uint8)

    n_tile = min(N_TILE_STIMA, len(files))
    indici = np.unique(np.linspace(0, len(files) - 1, n_tile).astype(int))

    for i in indici:
        img = cv2.imread(os.path.join(images_src, files[i]))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        px = img_rgb.reshape(-1, 3)

        # (a) campione GREZZO, vetro incluso -> serve per I0
        n_grezzo = min(N_PIXEL_I0_PER_TILE, px.shape[0])
        idx_g = normalizer.rng.choice(px.shape[0], n_grezzo, replace=False)
        pool_grezzo.append(px[idx_g])

        # (b) campione di solo TESSUTO -> serve per i vettori di stain
        mask = np.mean(px, axis=-1) < SOGLIA_TESSUTO_GREZZA
        px_tessuto = px[mask]
        if px_tessuto.shape[0] > n_per_tile:
            idx = normalizer.rng.choice(px_tessuto.shape[0], n_per_tile, replace=False)
            px_tessuto = px_tessuto[idx]
        if px_tessuto.shape[0] > 0:
            pool_tessuto.append(px_tessuto)

    vuoto = np.empty((0, 3), dtype=np.uint8)
    return (np.vstack(pool_tessuto) if pool_tessuto else vuoto,
            np.vstack(pool_grezzo) if pool_grezzo else vuoto)

def process_dataset():
    print("Avvio Preprocessing Vahadane (Modalità 1 - Tinta e I0 Dinamico)")
    normalizer = Vahadane(seed=SEED)
    
    wsi_folders = sorted([f for f in os.listdir(DATASET_DIR) 
                          if os.path.isdir(os.path.join(DATASET_DIR, f)) and not f.startswith("_")])

    stime_wsi = {}
    matrici_valide = []
    report_wsi = {}
    warnings = []
    
    # ------------------------------------------------------------
    # PASSATA A: Stima e Fallback Automatico
    # ------------------------------------------------------------
    for wsi in tqdm(wsi_folders, desc="Passata A - Estrazione e Controllo"):
        images_src = os.path.join(DATASET_DIR, wsi, CARTELLA_INPUT)
        if not os.path.exists(images_src):
            continue
            
        files = sorted([f for f in os.listdir(images_src) if f.endswith(ESTENSIONE)])
        if not files:
            continue

        pool_tessuto, pool_grezzo = raccogli_pixel_wsi(images_src, files, normalizer)
        HE_src, I0_wsi, status = normalizer.fit_wsi_matrix(pool_tessuto, pool_grezzo)

        fallback_applicato = False
        cos_h = cos_e = None
        if HE_src is None:
            HE_src = PRIOR_HE.copy()
            fallback_applicato = True
            warnings.append(f"[{wsi}] Fallback automatico: {status}")
        else:
            sim = np.sum(HE_src * PRIOR_HE, axis=0)
            cos_h, cos_e = float(sim[0]), float(sim[1])
            if min(cos_h, cos_e) < SOGLIA_ALLARME_COSENO:
                HE_src = PRIOR_HE.copy()
                fallback_applicato = True
                warnings.append(f"[{wsi}] Fallback automatico: Matrice anomala "
                                f"(cos_H={cos_h:.3f}, cos_E={cos_e:.3f})")
            else:
                matrici_valide.append(HE_src)

        stime_wsi[wsi] = {
            "HE_src": HE_src,
            "I0_wsi": I0_wsi,
            "files": files
        }

        # cos_H / cos_E vanno salvati SEMPRE, anche quando la stima e' stata
        # scartata: sono i valori che permettono di rianalizzare a posteriori
        # i casi borderline (quelli appena sopra soglia, che non generano
        # avvisi ma possono comunque meritare un controllo).
        report_wsi[wsi] = {
            "fallback_applicato": fallback_applicato,
            "status_stima": status,
            "cos_H_stimato": cos_h,
            "cos_E_stimato": cos_e,
            "HE_usata": HE_src.tolist(),
            "I0_calcolato": I0_wsi.tolist(),
            "n_tile_totali": len(files)
        }

    # ------------------------------------------------------------
    # PASSATA B: Calcolo Target Interno (Mediana Normalizzata)
    # ------------------------------------------------------------
    if not matrici_valide:
        raise RuntimeError("Nessuna matrice valida trovata per calcolare la mediana!")
        
    HE_target = np.median(np.stack(matrici_valide), axis=0)
    HE_target /= np.linalg.norm(HE_target, axis=0) # Normalizzazione necessaria!
    
    print("\nMatrice Target Calcolata (Mediana del Dataset):")
    print(np.round(HE_target, 4))
    print("-" * 50)

    # ------------------------------------------------------------
    # PASSATA C: Trasformazione
    # ------------------------------------------------------------
    for wsi, dati in tqdm(stime_wsi.items(), desc="Passata C - Normalizzazione"):
        images_src = os.path.join(DATASET_DIR, wsi, CARTELLA_INPUT)
        images_dst = os.path.join(DATASET_DIR, wsi, CARTELLA_OUTPUT)

        if os.path.exists(images_dst):
            shutil.rmtree(images_dst)
        os.makedirs(images_dst)

        HE_src = dati["HE_src"]
        I0_wsi = dati["I0_wsi"]

        for filename in dati["files"]:
            img_bgr = cv2.imread(os.path.join(images_src, filename))
            if img_bgr is None: continue
                
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            norm_rgb = normalizer.transform_tile(img_rgb, HE_src, I0_wsi, HE_target)
            
            norm_bgr = cv2.cvtColor(norm_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(images_dst, filename), norm_bgr)

    # ------------------------------------------------------------
    # SALVATAGGIO REPORT
    # ------------------------------------------------------------
    report_finale = {
        "HE_Target_Mediana": HE_target.tolist(),
        "parametri": {
            "alpha_sparsita": ALPHA_SPARSITA,
            "max_iter_nmf": MAX_ITER_NMF,
            "tolleranza_errore_nmf": TOLLERANZA_ERRORE_NMF,
            "max_pixel_fit": MAX_PIXEL_FIT,
            "n_tile_stima": N_TILE_STIMA,
            "soglia_allarme_coseno": SOGLIA_ALLARME_COSENO,
            "seed": SEED,
        },
        "n_wsi_con_stima_valida": len(matrici_valide),
        "wsi": report_wsi,
        "warnings": warnings
    }
    path_report = os.path.join(DATASET_DIR, NOME_REPORT)
    with open(path_report, "w", encoding="utf-8") as f:
        json.dump(report_finale, f, indent=2, ensure_ascii=False)
        
    print(f"\nPreprocessing completato! Report salvato in: {path_report}")
    if warnings:
        print("\nAvvisi di fallback risolti in automatico:")
        for w in warnings:
            print("   " + w)

if __name__ == "__main__":
    process_dataset()