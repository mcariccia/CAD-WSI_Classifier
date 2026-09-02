"""
metriche.py
"""
import numpy as np
import torch
import config as C

def nuova_cm(n=None, device=None):
    n = n or C.NUM_CLASSES
    return torch.zeros(n, n, dtype=torch.long, device=device)

def aggiorna_cm(cm, pred, target, n=None):
    n = n or C.NUM_CLASSES
    valido = target < n
    if valido.sum() == 0:
        return
    cm += torch.bincount(target[valido].reshape(-1) * n + pred[valido].reshape(-1),
                         minlength=n ** 2).reshape(n, n)

def aggiorna_cm_tile(cm_tile, pred, target, n=None, classe_sfondo=None):
    n = n or C.NUM_CLASSES
    classe_sfondo = C.CLASSE_SFONDO if classe_sfondo is None else classe_sfondo

    B = pred.shape[0]
    p = pred.reshape(B, -1)
    t = target.reshape(B, -1)

    tessuto = (t < n) & (t != classe_sfondo)
    peso = tessuto.to(torch.float32)

    t_safe = torch.where(tessuto, t, torch.zeros_like(t))

    conteggi_t = torch.zeros(B, n, device=pred.device, dtype=torch.float32)
    conteggi_p = torch.zeros(B, n, device=pred.device, dtype=torch.float32)
    conteggi_t.scatter_add_(1, t_safe, peso)
    conteggi_p.scatter_add_(1, p, peso)

    ha_tessuto = tessuto.any(dim=1)
    if not bool(ha_tessuto.any()):
        return

    vera = conteggi_t.argmax(1)[ha_tessuto]
    prev = conteggi_p.argmax(1)[ha_tessuto]
    cm_tile += torch.bincount(vera * n + prev,
                              minlength=n * n).reshape(n, n).to(cm_tile.dtype)
    

def cm_to_last(cm):
    if C.NUM_CLASSES == 4:
        return cm.clone()
    out = torch.zeros(4, 4, dtype=cm.dtype, device=cm.device)
    for gi, gruppo_i in enumerate(C.GRUPPI_LAST):
        for gj, gruppo_j in enumerate(C.GRUPPI_LAST):
            out[gi, gj] = cm[gruppo_i][:, gruppo_j].sum()
    return out

def iou_per_classe(cm):
    cm = cm.double()
    inter = torch.diag(cm)
    unione = cm.sum(0) + cm.sum(1) - inter
    return torch.where(unione > 0, inter / unione, torch.full_like(unione, float("nan")))

def dice_per_classe(cm):
    cm = cm.double()
    inter = torch.diag(cm)
    den = cm.sum(0) + cm.sum(1)
    return torch.where(den > 0, 2 * inter / den, torch.full_like(den, float("nan")))

def macro(v, escludi=()):
    keep = [k for k in range(len(v)) if k not in escludi]
    x = v[keep]
    x = x[~torch.isnan(x)]
    return float(x.mean()) if len(x) else float("nan")

def macro_presenti(cm, escludi=()):
    cm = cm.double()
    presenti = cm.sum(1) > 0
    for c in escludi:
        if c < len(presenti):
            presenti[c] = False
    if presenti.sum() == 0:
        return float("nan")
    inter = torch.diag(cm)
    unione = cm.sum(0) + cm.sum(1) - inter
    iou = torch.where(unione > 0, inter / unione, torch.zeros_like(unione))
    return float(iou[presenti].mean())

def accuratezza(cm):
    tot = float(cm.sum())
    return float(torch.diag(cm).sum()) / tot if tot > 0 else float("nan")

def accuratezza_bilanciata(cm, escludi=()):
    cm = cm.double()
    righe = cm.sum(1)
    presenti = righe > 0
    for c in escludi:
        if c < len(presenti):
            presenti[c] = False
    if presenti.sum() == 0:
        return float("nan")
    return float((torch.diag(cm)[presenti] / righe[presenti]).mean())

def recall_per_classe(cm):
    cm = cm.double()
    righe = cm.sum(1)
    return [float(cm[i, i] / righe[i]) if righe[i] > 0 else None for i in range(cm.shape[0])]

def metriche_hsil(cm_last):
    cm = cm_last.double()
    tessuto = [i for i in range(4) if i != C.IDX_GLASS_LAST]
    non_hsil = [i for i in tessuto if i != C.IDX_HSIL_LAST]
    tp = float(cm[C.IDX_HSIL_LAST, C.IDX_HSIL_LAST])
    fn = float(cm[C.IDX_HSIL_LAST, :].sum()) - tp
    fp = float(cm[non_hsil, C.IDX_HSIL_LAST].sum())
    tn = float(cm[non_hsil, :].sum()) - fp
    px_verso_vetro = float(cm[tessuto, C.IDX_GLASS_LAST].sum())
    nan = float("nan")
    ha_hsil = (tp + fn) > 0
    ha_non_hsil = (tn + fp) > 0
    predetti_hsil = (tp + fp) > 0
    return {
        "sensibilita": tp / (tp + fn) if ha_hsil else nan,
        "specificita": tn / (tn + fp) if ha_non_hsil else nan,
        "precisione": tp / (tp + fp) if predetti_hsil else nan,
        "dice": 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else nan,
        "iou": tp / (tp + fp + fn) if (tp + fp + fn) > 0 else nan,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "px_tessuto_predetti_vetro": px_verso_vetro,
    }

def discriminazione_lsil_hsil(cm_last):
    cm = cm_last.double()
    a, b = C.IDX_LSIL_LAST, C.IDX_HSIL_LAST
    ll, lh = float(cm[a, a]), float(cm[a, b])
    hl, hh = float(cm[b, a]), float(cm[b, b])
    tot = ll + lh + hl + hh
    nan = float("nan")
    return {
        "n_pixel_lesionali": tot,
        "accuratezza": (ll + hh) / tot if tot > 0 else nan,
        "recall_lsil": ll / (ll + lh) if (ll + lh) > 0 else nan,
        "recall_hsil": hh / (hl + hh) if (hl + hh) > 0 else nan,
    }

# --- AGGIUNTE DA LUCA: POSITIVO VS NEGATIVO ---
def cm_to_positivo_negativo(cm_last):
    cm = cm_last.double()
    G, N = C.IDX_GLASS_LAST, C.IDX_NEGATIVE_LAST
    L, H = C.IDX_LSIL_LAST, C.IDX_HSIL_LAST
    out = torch.zeros(2, 2, dtype=cm.dtype)
    out[0, 0] = cm[N, G] + cm[N, N]
    out[0, 1] = cm[N, L] + cm[N, H]
    out[1, 0] = cm[L, G] + cm[L, N] + cm[H, G] + cm[H, N]
    out[1, 1] = cm[L, L] + cm[L, H] + cm[H, L] + cm[H, H]
    return out

def metriche_positivo(cm_pn):
    tn, fp, fn, tp = (float(cm_pn[0, 0]), float(cm_pn[0, 1]),
                      float(cm_pn[1, 0]), float(cm_pn[1, 1]))
    nan = float("nan")
    return {
        "sensibilita": tp / (tp + fn) if (tp + fn) > 0 else nan,
        "specificita": tn / (tn + fp) if (tn + fp) > 0 else nan,
        "precisione": tp / (tp + fp) if (tp + fp) > 0 else nan,
        "dice": 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else nan,
        "iou": tp / (tp + fp + fn) if (tp + fp + fn) > 0 else nan,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }

def stampa_confusione(cm, nomi, titolo=""):
    cm = cm.double().cpu().numpy()
    norm = 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)
    if titolo:
        print(f"   {titolo}")
    intestazione = "vera \\ pred"        
    print(f"   {intestazione:<12}" + "".join(f"{n[:9]:>11}" for n in nomi))
    for i, nome in enumerate(nomi):
        riga = "".join(f"{norm[i, j]:>11.1f}" for j in range(len(nomi)))
        print(f"   {nome:<12}{riga}")

def stampa_per_classe(cm, nomi, titolo="IoU, Dice e recall per classe"):
    iou = iou_per_classe(cm)
    dice = dice_per_classe(cm)
    rec = recall_per_classe(cm)
    n = cm.double().sum(1)
    print(f"   {titolo}")
    print(f"   {'classe':<12}{'n px veri':>15}{'IoU':>9}{'Dice':>9}{'recall':>9}")
    for i, nome in enumerate(nomi):
        v_iou, v_dice = iou[i].item(), dice[i].item()
        s_iou = "n/a" if np.isnan(v_iou) else f"{v_iou:.4f}"
        s_dice = "n/a" if np.isnan(v_dice) else f"{v_dice:.4f}"
        s_rec = "n/a" if rec[i] is None else f"{rec[i]:.4f}"
        print(f"   {nome:<12}{int(n[i]):>15,}{s_iou:>9}{s_dice:>9}{s_rec:>9}")

def errori_principali(cm, nomi, top=8, escludi_diagonale=True):
    cm = cm.double().cpu().numpy()
    righe = cm.sum(1)
    voci = []
    for i in range(cm.shape[0]):
        if righe[i] == 0:
            continue
        for j in range(cm.shape[1]):
            if escludi_diagonale and i == j:
                continue
            n = float(cm[i, j])
            if n == 0:
                continue
            voci.append({
                "vera": nomi[i],
                "predetta": nomi[j],
                "quota": n / righe[i],
                "pixel": n,
                "quota_totale": n / max(cm.sum(), 1),
            })
    voci.sort(key=lambda v: -v["quota"])
    return voci[:top]

def stampa_errori_principali(cm, nomi, top=8, titolo="Errori principali"):
    voci = errori_principali(cm, nomi, top)
    if not voci:
        print(f"   {titolo}: nessun errore fuori diagonale.")
        return voci
    print(f"   {titolo} (ordinati per quota della classe vera)")
    print(f"   {'vera -> predetta':<28}{'quota':>9}{'pixel':>14}{'% totale':>10}")
    for v in voci:
        coppia = f"{v['vera']} -> {v['predetta']}"
        print(f"   {coppia:<28}{100*v['quota']:>8.1f}%{int(v['pixel']):>14,}"
              f"{100*v['quota_totale']:>9.2f}%")
    return voci