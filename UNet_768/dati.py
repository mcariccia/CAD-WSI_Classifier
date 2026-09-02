"""
dati.py
====================================================================
Lettura delle maschere, dataset, augmentation, composizione, fold,
sampler.

Quattro punti delicati, tutti documentati sul posto:
  - le maschere vanno lette con PIL, MAI con cv2 (palette vs indici)
  - il riempimento dell'augmentation va VERIFICATO, non assunto
  - i fold vanno bilanciati sulla presenza delle classi nelle WSI, non
    sul numero di tile
  - le zone non annotate NON sono tutte tessuto: marca_vetro le divide
    in vetro e tessuto prima dell'augmentation

NOTA. L'augmentation di colorazione vive in hed_augment.py e qui viene
solo importata. Non ridefinirla in questo file: una definizione locale
oscurerebbe l'import e le modifiche a hed_augment.py non avrebbero piu'
alcun effetto, mentre il training continuerebbe a girare producendo
numeri plausibili. E' gia' successo due volte.

CutMix e' stato rimosso: il taglio rettangolare recide l'asse
basale-superficie dell'epitelio, che e' l'asse su cui e' definito il
criterio diagnostico LAST (frazione di spessore occupata da cellule
basaloidi immature). Produceva epiteli morfologicamente impossibili.
====================================================================
"""

import os
import json
import random
import warnings
from collections import defaultdict, Counter

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from hed_augment import HEDIntensita

import config as C

Image.MAX_IMAGE_PIXELS = None

# LUT di rimappatura su LAST, usata solo in MODALITA_CLASSI = 4
_LUT_RIMAPPA = None
if C.RIMAPPA_MASCHERE:
    _LUT_RIMAPPA = np.full(256, C.IGNORE_INDEX, dtype=np.uint8)
    for _c in range(C.N_CLASSI_FINI):
        _LUT_RIMAPPA[_c] = C.LUT_LAST_FINI[_c]


# ====================================================================
# 1. LETTURA
# ====================================================================
def leggi_maschera(path):
    """
    Le maschere prodotte da LabeledImageServer sono PNG INDICIZZATI.
    cv2.imread applicherebbe la palette restituendo la LUMINANZA del
    colore (CIN1 -> 212, HSIL -> 29, ...) invece dell'ID di classe, e
    senza segnalarlo. PIL restituisce gli indici.
    """
    im = Image.open(path)
    a = np.array(im)
    if a.ndim != 2:
        raise RuntimeError(
            f"{path}: maschera con {a.ndim} dimensioni (mode='{im.mode}'). "
            f"Attesa immagine a indici.")
    if _LUT_RIMAPPA is not None:
        a = _LUT_RIMAPPA[a]
    return a


def elenca_campioni(dataset_dir=None):
    """-> lista di (path_immagine, path_maschera, nome_wsi)"""
    dataset_dir = dataset_dir or C.DATASET_DIR
    campioni, mancanti = [], []
    wsi_folders = sorted(f for f in os.listdir(dataset_dir)
                         if os.path.isdir(os.path.join(dataset_dir, f))
                         and not f.startswith("_"))
    for wsi in wsi_folders:
        d_img = os.path.join(dataset_dir, wsi, C.CARTELLA_IMG)
        d_msk = os.path.join(dataset_dir, wsi, C.CARTELLA_MSK)
        if not os.path.isdir(d_img):
            mancanti.append(wsi)
            continue
        for f in sorted(os.listdir(d_img)):
            if not f.endswith(C.ESTENSIONI):
                continue
            p_msk = os.path.join(d_msk, f)
            if os.path.exists(p_msk):
                campioni.append((os.path.join(d_img, f), p_msk, wsi))
    if mancanti:
        print(f"   ATTENZIONE: cartella '{C.CARTELLA_IMG}' assente in "
              f"{len(mancanti)} WSI: {', '.join(mancanti[:5])}"
              f"{' ...' if len(mancanti) > 5 else ''}")
    return campioni


# ====================================================================
# 2. PSEUDO-ETICHETTA VETRO / TESSUTO
# ====================================================================
def marca_vetro(image, mask, soglia_od=0.08, ignore_index=255,
                valore_vetro=254):
    """
    I pixel non annotati non sono privi di etichetta rispetto alla
    dicotomia vetro/tessuto: lo sono solo rispetto alla CLASSE
    DIAGNOSTICA. Il vetro in H&E si separa con una soglia sulla densita'
    ottica, quindi si distingue ignore_index (non annotato ma tessuto)
    da valore_vetro (non annotato e vetro).

    Perche' dentro la maschera e non in un secondo tensore: cosi' la
    pseudo-etichetta attraversa le stesse trasformazioni geometriche con
    l'interpolazione nearest gia' verificata, il fill del padding resta
    coerente, e la firma del DataLoader non cambia.

    Si calcola sull'immagine ORIGINALE, prima dell'augmentation: la
    pseudo-etichetta descrive il vetrino, non la sua versione perturbata.
    Altrimenti un'attenuazione forte dell'ematossilina trasformerebbe
    tessuto pallido in "vetro", cioe' esattamente l'errore da correggere.
    """
    ignoti = mask == ignore_index
    if not ignoti.any():
        return mask
    od = -np.log(np.clip(image.astype(np.float32) / 255.0, 1e-3, 1.0)).mean(2)
    fuori = mask.copy()
    fuori[ignoti & (od < soglia_od)] = np.uint8(valore_vetro)
    return fuori


# ====================================================================
# 3. AUGMENTATION
# ====================================================================
_KWARGS_RIEMPIMENTO = None

# Riempimento delle zone vuote create dalle trasformazioni geometriche.
VALORE_FILL_IMG = 255                 # bianco: in H&E il vuoto e' vetro
VALORE_FILL_MSK = C.CLASSE_SFONDO     # vetro, coerente con il bianco

_CLASSE_PROVA = next(c for c in range(C.NUM_CLASSES) if c != C.CLASSE_SFONDO)


def _prova_riempimento(kw):
    img = np.full((64, 64, 3), 30, dtype=np.uint8)
    msk = np.full((64, 64), _CLASSE_PROVA, dtype=np.uint8)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = A.Affine(rotate=(45, 45), p=1.0, **kw)(image=img, mask=msk)
    except Exception:
        return False
    msk_ok = VALORE_FILL_MSK in set(np.unique(out["mask"]).tolist())
    img_ok = VALORE_FILL_IMG in set(np.unique(out["image"]).tolist())
    return bool(msk_ok and img_ok)


def kwargs_riempimento():
    global _KWARGS_RIEMPIMENTO
    if _KWARGS_RIEMPIMENTO is not None:
        return _KWARGS_RIEMPIMENTO
    candidati = [
        ("albumentations >= 2.0",
         dict(border_mode=cv2.BORDER_CONSTANT, fill=VALORE_FILL_IMG,
              fill_mask=VALORE_FILL_MSK)),
        ("albumentations 1.4-2.0",
         dict(fill=VALORE_FILL_IMG, fill_mask=VALORE_FILL_MSK)),
        ("albumentations 1.x",
         dict(mode=cv2.BORDER_CONSTANT, cval=VALORE_FILL_IMG,
              cval_mask=VALORE_FILL_MSK)),
    ]
    for etichetta, kw in candidati:
        if _prova_riempimento(kw):
            _KWARGS_RIEMPIMENTO = kw
            print(f"   API riempimento rilevata: {etichetta}")
            return kw
    raise RuntimeError("Nessuna combinazione applica il riempimento...")


def affine():
    return A.Affine(translate_percent=(-0.05, 0.05), scale=(0.85, 1.15),
                    rotate=(-15, 15), p=0.5, **kwargs_riempimento())


def lista_augmentation(usa_hed=None, par_hed=None):
    """
    UNICA definizione. I parametri arrivano da config.parametri_hed(),
    che e' anche cio' che finisce nel report: i valori effettivi del run
    non possono divergere da quelli documentati.
    """
    usa_hed = C.USA_HED if usa_hed is None else usa_hed
    par_hed = C.parametri_hed() if par_hed is None else par_hed
    aug = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=1.0),
        affine(),
        A.GaussianBlur(blur_limit=(3, 5), p=0.20),
        A.GaussNoise(p=0.15),
    ]
    if usa_hed:
        aug.append(HEDIntensita(**par_hed))
    return aug


def transforms_train(usa_hed=None, par_hed=None):
    return A.Compose(lista_augmentation(usa_hed, par_hed)
                     + [A.Normalize(), ToTensorV2()])


def transforms_val():
    return A.Compose([A.Normalize(), ToTensorV2()])


def init_worker(worker_id):
    """
    Ri-semina numpy in ogni worker del DataLoader.
    HEDIntensita usa np.random direttamente: con questa funzione ogni
    worker riceve un seme distinto ma deterministico, derivato dal seme
    base dell'epoca.
    """
    seme = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seme)
    random.seed(seme)


# ====================================================================
# 4. DATASET
# ====================================================================
class DatasetPrivato(Dataset):
    def __init__(self, campioni, transform=None, par_vetro=None):
        self.campioni = campioni
        self.transform = transform
        self.par_vetro = C.parametri_vetro() if par_vetro is None else par_vetro

    def __len__(self):
        return len(self.campioni)

    def __getitem__(self, idx):
        p_img, p_msk, wsi = self.campioni[idx]
        bgr = cv2.imread(p_img)
        if bgr is None:
            raise RuntimeError(f"immagine illeggibile: {p_img}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = marca_vetro(image, leggi_maschera(p_msk), **self.par_vetro)
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        return image.float(), mask.long(), wsi


# ====================================================================
# 5. COMPOSIZIONE
# ====================================================================
def composizione(campioni, cache_path=None):
    """
    Conteggio dei pixel per classe di ogni tile. Serve a fold e sampler.

    Legge le maschere da disco, dove esistono solo 0..NUM_CLASSES-1 e
    255: il 254 nasce a runtime in marca_vetro e non entra qui. Il
    filtro vv < NUM_CLASSES esclude comunque entrambi.
    """
    cache_path = cache_path or os.path.join(C.DIR_CACHE, "composizione.json")
    firma = f"{len(campioni)}|{C.NUM_CLASSES}|{campioni[0][0]}"
    if os.path.exists(cache_path):
        try:
            d = json.load(open(cache_path))
            if d.get("firma") == firma:
                print("   (composizione letta da cache)")
                return np.array(d["comp"], dtype=np.float64)
        except Exception:
            pass

    comp = np.zeros((len(campioni), C.NUM_CLASSES), dtype=np.float64)
    for i, (_, p_msk, _) in enumerate(tqdm(campioni, desc="   composizione",
                                           ncols=78)):
        m = leggi_maschera(p_msk)
        v, c = np.unique(m, return_counts=True)
        for vv, cc in zip(v, c):
            if vv < C.NUM_CLASSES:
                comp[i, vv] = cc
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        json.dump({"firma": firma, "comp": comp.tolist()}, open(cache_path, "w"))
    except Exception:
        pass
    return comp


def pixel_per_wsi(campioni, comp):
    px = defaultdict(lambda: np.zeros(C.NUM_CLASSES))
    for i, (_, _, w) in enumerate(campioni):
        px[w] += comp[i]
    return dict(px)


def presenza_classi(px_wsi, soglia=None):
    """-> {wsi: set(classi presenti sopra soglia)}"""
    soglia = C.SOGLIA_PRESENZA_CLASSE if soglia is None else soglia
    out = {}
    for w, v in px_wsi.items():
        tot = max(v.sum(), 1.0)
        out[w] = {c for c in range(C.NUM_CLASSES) if v[c] / tot >= soglia}
    return out


# ====================================================================
# 6. FOLD
# ====================================================================
def crea_fold(campioni, presenza, n_fold=None, seed=None):
    """
    Fold bilanciati sulla PRESENZA DELLE CLASSI nelle WSI, non sul
    numero di tile.

    Perche' non basta la classe dominante. Su MTCHI ogni paziente aveva
    di fatto una classe sola e bastava bilanciare quella. Qui una WSI
    contiene piu' classi insieme (Ghiandole in 27 WSI, Mucosa in 22,
    HSIL in 22, CIN1 in 10, Stroma in 8), quindi va bilanciata la
    presenza di CIASCUNA classe, altrimenti una classe rara finisce
    concentrata in un fold e assente dagli altri.

    Algoritmo: stratificazione iterativa (Sechidis et al. 2011,
    semplificata). Si processano le classi dalla piu' rara alla piu'
    comune; per ciascuna si assegnano le WSI ancora libere al fold che
    ne ha meno. Una WSI gia' assegnata non si sposta piu': e' cio' che
    garantisce l'assenza di leakage.

    A seme fisso l'assegnazione e' deterministica: i fold sono identici
    fra run diverse, quindi i confronti dell'ablation sono appaiati.
    """
    n_fold = C.N_FOLD if n_fold is None else n_fold
    seed = C.SEED if seed is None else seed
    wsi_list = sorted(presenza.keys())
    rng = random.Random(seed)

    n_per_classe = {c: sum(1 for s in presenza.values() if c in s)
                    for c in range(C.NUM_CLASSES)}
    ordine = sorted(range(C.NUM_CLASSES), key=lambda c: n_per_classe[c])

    assegnazione = {}
    conteggio = [defaultdict(int) for _ in range(n_fold)]
    n_wsi_fold = [0] * n_fold

    for c in ordine:
        gruppo = [w for w in wsi_list
                  if c in presenza[w] and w not in assegnazione]
        rng.shuffle(gruppo)
        for w in gruppo:
            k = min(range(n_fold),
                    key=lambda j: (conteggio[j][c], n_wsi_fold[j], j))
            assegnazione[w] = k
            n_wsi_fold[k] += 1
            for cc in presenza[w]:
                conteggio[k][cc] += 1

    for w in wsi_list:                    # WSI senza classi sopra soglia
        if w not in assegnazione:
            k = min(range(n_fold), key=lambda j: (n_wsi_fold[j], j))
            assegnazione[w] = k
            n_wsi_fold[k] += 1

    folds = []
    for k in range(n_fold):
        wsi_val = {w for w in wsi_list if assegnazione[w] == k}
        va = np.array([i for i, (_, _, w) in enumerate(campioni) if w in wsi_val])
        tr = np.array([i for i, (_, _, w) in enumerate(campioni)
                       if w not in wsi_val])
        folds.append((tr, va))
    return folds, assegnazione


def stampa_fold(folds, assegnazione, presenza):
    print("\n  --- COMPOSIZIONE DEI FOLD ---")
    print(f"  {'fold':>5} | {'WSI':>4} | {'tile':>7} | classi presenti in validation")
    print("  " + "-" * 74)
    for k, (_, va) in enumerate(folds):
        wsi_val = sorted(w for w, kk in assegnazione.items() if kk == k)
        cls = sorted({c for w in wsi_val for c in presenza[w]})
        mancanti = [C.CLASS_NAMES[c] for c in range(C.NUM_CLASSES) if c not in cls]
        print(f"  {k:>5} | {len(wsi_val):>4} | {len(va):>7} | "
              f"{', '.join(C.CLASS_NAMES[c] for c in cls)}")
        if mancanti:
            print(f"  {'':>5} | {'':>4} | {'':>7} |   assenti: "
                  f"{', '.join(mancanti)}  <-- non misurate in questo fold")
    quote = [len(va) / sum(len(v) for _, v in folds) for _, va in folds]
    print(f"  quota di tile per fold: min {min(quote):.1%}  max {max(quote):.1%}")


# ====================================================================
# 7. SAMPLER E PESI
# ====================================================================
def crea_sampler(comp_tr, wsi_tr):
    """
    Peso di una tile = rarita' media delle classi che contiene, diviso
    per il numero di tile della sua WSI elevato a WSI_BALANCE.

    Lo Sfondo (vetro) e' escluso dal calcolo della rarita': e' banale da
    imparare, e bilanciarlo penalizzerebbe le tile che contengono
    tessuti rari sui bordi.
    """
    freq = comp_tr.sum(0)

    inv = np.zeros_like(freq)
    for c in range(C.NUM_CLASSES):
        if c == C.CLASSE_SFONDO:
            inv[c] = 0.0
        else:
            inv[c] = (1.0 / max(freq[c], 1e-8)) ** C.SAMPLING_ALPHA

    somma_inv = inv.sum()
    if somma_inv > 0:
        inv /= somma_inv

    n_per_wsi = Counter(wsi_tr)
    pesi = []
    for i, w in enumerate(wsi_tr):
        tot_tessuto = comp_tr[i, 1:].sum()
        if tot_tessuto > 0:
            p = comp_tr[i] / tot_tessuto
            p[C.CLASSE_SFONDO] = 0.0
            rarita = max(float((p * inv).sum()), 1e-8)
        else:
            rarita = 1e-8
        pesi.append(rarita * (1.0 / n_per_wsi[w]) ** C.WSI_BALANCE)

    pesi = torch.DoubleTensor(pesi)
    print("   composizione train: " +
          "  ".join(f"{n} {100*(f/max(freq.sum(),1)):.1f}%"
                    for n, f in zip(C.CLASS_NAMES, freq)))
    print(f"   pesi sampler: rapporto max/min {pesi.max()/pesi.min():.1f}x")
    return WeightedRandomSampler(pesi, num_samples=len(pesi),
                                 replacement=True), freq


def pesi_loss(freq):
    """
    Frequenza inversa con CLIP. Il clip e' essenziale: il sampler sta
    gia' correggendo lo sbilanciamento, e due correzioni aggressive in
    serie fanno sovra-predire le classi rare ovunque (recall alta,
    precision a terra).
    """
    f = np.maximum(np.asarray(freq, dtype=np.float64), 1e-8)
    w = np.clip(np.median(f) / f, *C.CLIP_PESI)
    print("   pesi di loss:", {n: round(float(v), 2)
                               for n, v in zip(C.CLASS_NAMES, w)})
    return torch.tensor(w, dtype=torch.float32)


def crea_loader(campioni, idx_tr, idx_va, comp):
    c_tr = [campioni[i] for i in idx_tr]
    c_va = [campioni[i] for i in idx_va]
    wsi_tr = [c[2] for c in c_tr]
    wsi_va = sorted({c[2] for c in c_va})

    if set(wsi_tr) & set(wsi_va):
        raise RuntimeError("LEAKAGE: WSI presente in train e validation")
    if len(c_tr) < C.BATCH_SIZE:
        raise RuntimeError(
            f"Solo {len(c_tr)} tile in training, meno del batch "
            f"({C.BATCH_SIZE}): con drop_last=True il loader sarebbe vuoto.")
    if len(c_va) == 0:
        raise RuntimeError("Nessuna tile in validation per questo fold.")

    train_ds = DatasetPrivato(c_tr, transforms_train())
    val_ds = DatasetPrivato(c_va, transforms_val())

    sampler, freq = crea_sampler(comp[idx_tr], wsi_tr)
    comune = dict(num_workers=C.NUM_WORKERS, pin_memory=True,
                  persistent_workers=C.NUM_WORKERS > 0,
                  worker_init_fn=init_worker if C.NUM_WORKERS > 0 else None)

    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE,
                              drop_last=True,
                              **({"sampler": sampler} if C.USA_SAMPLER
                                 else {"shuffle": True}),
                              **comune)
    val_loader = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                            **comune)
    return train_loader, val_loader, freq, wsi_va