"""
dati.py
====================================================================
Lettura delle maschere, dataset, augmentation, composizione, fold,
sampler.

Tre punti delicati, tutti documentati sul posto:
  - le maschere vanno lette con PIL, MAI con cv2 (palette vs indici)
  - il riempimento dell'augmentation va VERIFICATO, non assunto
  - i fold vanno bilanciati sulla presenza delle classi nelle WSI, non
    sul numero di tile
====================================================================
"""

import os
import json
import math
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
# 2. AUGMENTATION
# ====================================================================

HE_CANONICA = np.array([[0.5626, 0.2159],
                        [0.7201, 0.8012],
                        [0.4062, 0.5581]], dtype=np.float32)
PINV_HE = np.linalg.pinv(HE_CANONICA).astype(np.float32)
I0_HED = np.float32(255.0)

class HEDIntensita(A.ImageOnlyTransform):
    """
    Augmentation moltiplicativa delle concentrazioni H&E.
    Preserva l'ipercromasia nucleare ed evita di colorare il vetro (sfondo).
    """
    def __init__(self, k_intensita=2.0, k_rapporto=1.3,
                 bias_relativo=0.02, scala_h=1.02, scala_e=0.25, p=0.8):
        super().__init__(p=p)
        self.log_k_int = float(np.log(max(k_intensita, 1.0 + 1e-9)))
        self.log_k_rap = float(np.log(max(k_rapporto, 1.0 + 1e-9)))
        self.ampiezza_bias = np.array([bias_relativo * scala_h,
                                       bias_relativo * scala_e],
                                      dtype=np.float32)

    @staticmethod
    def _scomponi(img):
        """RGB uint8 -> (concentrazioni (2,N), residuo (3,N), forma)."""
        h, w, _ = img.shape
        px = img.reshape(-1, 3).astype(np.float32)
        OD = -np.log((px + 1.0) / I0_HED).T
        C = np.maximum(PINV_HE @ OD, 0.0)
        residuo = OD - HE_CANONICA @ C
        return C, residuo, (h, w)

    @staticmethod
    def _ricomponi(C, residuo, forma):
        OD = HE_CANONICA @ C + residuo
        px = I0_HED * np.exp(-OD) - 1.0
        return np.clip(px.T, 0, 255).astype(np.uint8).reshape(*forma, 3)

    def apply(self, img, **params):
        C, residuo, forma = self._scomponi(img)

        g = np.exp(np.random.uniform(-self.log_k_int, self.log_k_int))
        r = np.exp(np.random.uniform(-self.log_k_rap, self.log_k_rap))
        alpha = np.array([[g * r], [g / r]], dtype=np.float32)
        beta = np.random.uniform(-self.ampiezza_bias,
                                 self.ampiezza_bias).reshape(2, 1)

        C = np.maximum(C * alpha + beta, 0.0)
        return self._ricomponi(C, residuo, forma)

    def get_transform_init_args_names(self):
        return ("log_k_int", "log_k_rap", "ampiezza_bias")


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

def lista_augmentation():
    aug = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=1.0),
        affine(),
        A.GaussianBlur(blur_limit=(3, 5), p=0.20),
        A.GaussNoise(p=0.15),
    ]
    if C.USA_HED:
        aug.append(HEDIntensita(
            k_intensita=C.K_INTENSITA,
            k_rapporto=C.K_RAPPORTO,
            bias_relativo=C.BIAS_REL,
            scala_h=C.SCALA_H,
            scala_e=C.SCALA_E,
            p=C.P_HED
        ))
    return aug

def transforms_train():
    return A.Compose(lista_augmentation() + [A.Normalize(), ToTensorV2()])

def transforms_val():
    return A.Compose([A.Normalize(), ToTensorV2()])

def cutmix(images, masks, p=None, beta=None):
    """
    Incolla un rettangolo casuale preso da un'altra tile del batch, sia
    nell'immagine sia nella maschera.

    PERCHE' SERVE QUI. Le tile del dataset privato sono quasi tutte mono
    o bi-classe, e le WSI stesse lo sono: nessuna tile contiene tutte le
    classi. Una rete addestrata cosi' puo' minimizzare la loss con una
    decisione GLOBALE (guarda la tile, scegli la classe, riempi) senza
    mai imparare a localizzare. CutMix crea tile in cui zone diverse
    hanno etichette diverse e costringe a decidere per pixel.

    LIMITE DA DICHIARARE: il confine generato e' un rettangolo, non una
    transizione morfologica. Insegna localita', non la morfologia del
    passaggio fra gradi — che nel dataset non e' presente.

    Le etichette si spostano coi pixel, quindi non serve mescolare la
    loss con lambda come nel CutMix per classificazione.
    """
    p = C.P_CUTMIX if p is None else p
    beta = C.CUTMIX_BETA if beta is None else beta
    if random.random() > p:
        return images, masks
    B, _, H, W = images.shape
    perm = torch.randperm(B, device=images.device)
    lam = np.random.beta(beta, beta)
    rh, rw = int(H * math.sqrt(1.0 - lam)), int(W * math.sqrt(1.0 - lam))
    if rh < 8 or rw < 8:
        return images, masks
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = max(0, cy - rh // 2), min(H, cy + rh // 2)
    x1, x2 = max(0, cx - rw // 2), min(W, cx + rw // 2)
    if y2 <= y1 or x2 <= x1:
        return images, masks

    # Si permuta SOLO la regione da incollare, non l'intero batch.
    # images[perm][:, :, y1:y2, x1:x2] materializzerebbe una copia di
    # tutto il batch (12 x 3 x 512 x 512 = 37 MB) per poi scartarne la
    # quasi totalita'. Indicizzando prima la regione, la copia e' grande
    # quanto il rettangolo.
    images[:, :, y1:y2, x1:x2] = images[:, :, y1:y2, x1:x2][perm]
    masks[:, y1:y2, x1:x2] = masks[:, y1:y2, x1:x2][perm]
    return images, masks


def init_worker(worker_id):
    """
    Ri-semina numpy in ogni worker del DataLoader.
    HEDJitterAsimmetrico usa np.random direttamente, con questa funzione ogni worker riceve un seme
    distinto ma deterministico, derivato dal seme base dell'epoca.
    """
    seme = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seme)
    random.seed(seme)


# ====================================================================
# 3. DATASET
# ====================================================================
class DatasetPrivato(Dataset):
    def __init__(self, campioni, transform=None):
        self.campioni = campioni
        self.transform = transform

    def __len__(self):
        return len(self.campioni)

    def __getitem__(self, idx):
        p_img, p_msk, wsi = self.campioni[idx]
        bgr = cv2.imread(p_img)
        if bgr is None:
            raise RuntimeError(f"immagine illeggibile: {p_img}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = leggi_maschera(p_msk)
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        return image.float(), mask.long(), wsi


# ====================================================================
# 4. COMPOSIZIONE
# ====================================================================
def composizione(campioni, cache_path=None):
    """Conteggio dei pixel per classe di ogni tile. Serve a fold e sampler."""
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
# 5. FOLD
# ====================================================================
def crea_fold(campioni, presenza, n_fold=None, seed=None):
    """
    Fold bilanciati sulla PRESENZA DELLE CLASSI nelle WSI, non sul
    numero di tile.

    Perche' non basta la classe dominante. Su MTCHI ogni paziente aveva
    di fatto una classe sola e bastava bilanciare quella. Qui una WSI
    contiene piu' classi insieme (Ghiandole in ~30 WSI, Mucosa in ~26,
    Stroma in ~10, CIN1 in ~12), quindi va bilanciata la presenza di
    CIASCUNA classe, altrimenti una classe rara finisce concentrata in
    un fold e assente dagli altri.

    Algoritmo: stratificazione iterativa (Sechidis et al. 2011,
    semplificata). Si processano le classi dalla piu' rara alla piu'
    comune; per ciascuna si assegnano le WSI ancora libere al fold che
    ne ha meno. Una WSI gia' assegnata non si sposta piu': e' cio' che
    garantisce l'assenza di leakage.
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
# 6. SAMPLER E PESI
# ====================================================================
def crea_sampler(comp_tr, wsi_tr):
    """
    Peso di una tile = rarita' media delle classi che contiene, diviso
    per il numero di tile della sua WSI elevato a WSI_BALANCE.

    Il secondo fattore e' essenziale su questo dataset: lo Stroma e'
    concentrato in 2 WSI su 40 che da sole valgono l'83% dei suoi pixel.
    Senza cap per WSI, quelle due dominerebbero ogni epoca.
    """
    freq = comp_tr.sum(0)
    freq = freq / max(freq.sum(), 1.0)
    inv = (1.0 / np.maximum(freq, 1e-8)) ** C.SAMPLING_ALPHA
    inv /= inv.sum()

    n_per_wsi = Counter(wsi_tr)
    pesi = []
    for i, w in enumerate(wsi_tr):
        tot = comp_tr[i].sum()
        p = comp_tr[i] / tot if tot > 0 else np.zeros(C.NUM_CLASSES)
        rarita = max(float((p * inv).sum()), 1e-8)
        pesi.append(rarita * (1.0 / n_per_wsi[w]) ** C.WSI_BALANCE)

    pesi = torch.DoubleTensor(pesi)
    print(f"   composizione train: " +
          "  ".join(f"{n} {100*f:.1f}%" for n, f in zip(C.CLASS_NAMES, freq)))
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