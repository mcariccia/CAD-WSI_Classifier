"""
metriche.py
====================================================================
Confusioni, IoU, collasso sulla gerarchia LAST, metriche cliniche.

Tre livelli di lettura, deliberatamente separati:

  1. CLASSI FINI  -> diagnostica: dove sbaglia il modello
  2. LAST         -> metrica di selezione: l'errore Stroma <-> Mucosa
                     non conta, sono entrambe negative
  3. HSIL vs non  -> deliverable clinico: soglia ASCCP 2019 fra
                     sorveglianza e trattamento escissionale
====================================================================
"""

import numpy as np
import torch

import config as C


# --------------------------------------------------------------------
# ACCUMULO
# --------------------------------------------------------------------
def nuova_cm(n=None, device=None):
    n = n or C.NUM_CLASSES
    return torch.zeros(n, n, dtype=torch.long, device=device)


def aggiorna_cm(cm, pred, target, n=None):
    n = n or C.NUM_CLASSES
    valido = target != C.IGNORE_INDEX
    if valido.sum() == 0:
        return
    cm += torch.bincount(target[valido].reshape(-1) * n + pred[valido].reshape(-1),
                         minlength=n ** 2).reshape(n, n)


def aggiorna_cm_tile(cm_tile, pred, target):
    """
    Confusione a livello di TILE: la classe "diagnostica" della tile
    contro quella predetta.

    IL VETRO E' ESCLUSO DALLA DEFINIZIONE DI CLASSE DELLA TILE.
    Se si prendesse la maggioranza su TUTTI i pixel annotati, in una
    tile meta' vetro e meta' tessuto la classe risulterebbe "Sfondo"
    sia nella verita' sia nella predizione, e l'accuratezza per tile
    varrebbe 1,00 anche con un modello inutile (verificato su dati
    sintetici: accuratezza 1,0000 con IoU per classe fra 0,015 e 0,40).

    Definizione corretta:
      - si considerano solo i pixel il cui valore VERO e' un tessuto
        (ne' vetro ne' ignore);
      - classe vera della tile   = maggioranza della verita' li';
      - classe predetta          = maggioranza della predizione li',
        includendo Sfondo fra le possibili risposte, cosi' che
        "tessuto predetto come vetro" compaia come errore.

    Le tile senza pixel di tessuto vengono saltate: non hanno una
    classe diagnostica.

    Serve perche' le annotazioni sono region-level: il patologo grada
    un campo, non un pixel. Questa metrica riproduce quel ragionamento.

    IMPLEMENTAZIONE VETTORIZZATA. La versione con ciclo Python lanciava
    2 x batch_size kernel CUDA minuscoli per ogni batch (con batch 12
    sono 24 lanci, ognuno con la propria latenza): il costo era
    dominato dall'overhead, non dal calcolo. Qui bastano due
    scatter_add_ e un bincount per l'intero batch.
    """
    B = pred.shape[0]
    K = C.NUM_CLASSES
    p = pred.reshape(B, -1)
    t = target.reshape(B, -1)

    tessuto = (t != C.IGNORE_INDEX) & (t != C.CLASSE_SFONDO)
    peso = tessuto.to(torch.float32)

    # Il peso vale 0 sui pixel non-tessuto, quindi il loro contributo e'
    # nullo qualunque indice si usi: basta rendere l'indice VALIDO.
    # 255 non e' un indice ammesso da scatter_add_, mentre pred contiene
    # sempre valori in [0, K-1] e puo' essere usato cosi' com'e'.
    t_safe = torch.where(tessuto, t, torch.zeros_like(t))

    conteggi_t = torch.zeros(B, K, device=pred.device, dtype=torch.float32)
    conteggi_p = torch.zeros(B, K, device=pred.device, dtype=torch.float32)
    conteggi_t.scatter_add_(1, t_safe, peso)
    conteggi_p.scatter_add_(1, p, peso)

    ha_tessuto = tessuto.any(dim=1)
    if not bool(ha_tessuto.any()):
        return
    vera = conteggi_t.argmax(1)[ha_tessuto]
    prev = conteggi_p.argmax(1)[ha_tessuto]
    cm_tile += torch.bincount(vera * K + prev,
                              minlength=K * K).reshape(K, K).to(cm_tile.dtype)


# --------------------------------------------------------------------
# COLLASSO SULLA GERARCHIA
# --------------------------------------------------------------------
def cm_to_last(cm):
    """
    Collassa una confusione fine sui 4 gruppi LAST sommando i blocchi.
    Conserva il totale dei pixel: nessuna informazione persa, solo
    riaggregata. L'errore Stroma -> Mucosa finisce sulla diagonale
    NEGATIVE, quindi smette di contare come errore.
    """
    if C.NUM_CLASSES == 4:
        return cm.clone()
    out = torch.zeros(4, 4, dtype=cm.dtype, device=cm.device)
    for gi, gruppo_i in enumerate(C.GRUPPI_LAST):
        for gj, gruppo_j in enumerate(C.GRUPPI_LAST):
            out[gi, gj] = cm[gruppo_i][:, gruppo_j].sum()
    return out


# --------------------------------------------------------------------
# INDICI
# --------------------------------------------------------------------
def iou_per_classe(cm):
    cm = cm.double()
    inter = torch.diag(cm)
    unione = cm.sum(0) + cm.sum(1) - inter
    return torch.where(unione > 0, inter / unione,
                       torch.full_like(unione, float("nan")))


def dice_per_classe(cm):
    cm = cm.double()
    inter = torch.diag(cm)
    den = cm.sum(0) + cm.sum(1)
    return torch.where(den > 0, 2 * inter / den,
                       torch.full_like(den, float("nan")))


def macro(v, escludi=()):
    keep = [k for k in range(len(v)) if k not in escludi]
    x = v[keep]
    x = x[~torch.isnan(x)]
    return float(x.mean()) if len(x) else float("nan")


def macro_presenti(cm, escludi=()):
    """
    Media delle IoU SOLO sulle classi presenti nella ground truth.

    In iou_per_classe una classe assente dal vero ma predetta anche in
    un solo pixel ha unione > 0 e intersezione 0, quindi IoU = 0, non
    NaN: macro() la conterebbe. Su una WSI che contiene 2 classi su 6,
    il massimo raggiungibile crollerebbe al primo falso positivo, e la
    metrica misurerebbe "quante classi spurie sono comparse" invece
    della qualita' della segmentazione.

    I falsi positivi restano puniti: entrano nell'unione delle classi
    presenti. Ma una classe che nella WSI non esiste non vale zero.
    """
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
    """
    Media delle recall per classe.

    L'accuratezza grezza e' dominata dalle WSI piu' grandi. Su MTCHI un
    paziente con 1540 tile su 2264 produceva accuratezza 0,90 mentre una
    classe era a ZERO. Con Stroma concentrato in 2 WSI su 40, qui il
    rischio e' identico.

    escludi: indici da NON mediare. A livello LAST va sempre passato
    GLASS: il vetro ha recall vicina a 1 quasi gratuitamente (e' bianco
    e occupa aree ampie e uniformi), quindi includerlo alzerebbe la
    media di un quarto senza dire nulla sulla capacita' diagnostica.
    """
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
    return [float(cm[i, i] / righe[i]) if righe[i] > 0 else None
            for i in range(cm.shape[0])]


# --------------------------------------------------------------------
# METRICHE CLINICHE
# --------------------------------------------------------------------
def metriche_hsil(cm_last):
    """
    HSIL vs non-HSIL: il deliverable primario.

    E' la soglia che decide la gestione clinica (ASCCP 2019, Perkins et
    al., J Low Genit Tract Dis 24:102-131): sopra si procede a
    escissione, sotto a sorveglianza.

    IL VETRO E' ESCLUSO DALLE RIGHE, NON DALLE COLONNE.

    Righe (verita'): si considerano solo i pixel di TESSUTO. I pixel di
    vetro sono banali da riconoscere e, contati come veri negativi,
    gonfierebbero la specificita' senza dire nulla sulla capacita'
    diagnostica.

    Colonne (predizione): si considerano TUTTE, vetro incluso. Un pixel
    di HSIL predetto come vetro E' un falso negativo — la lesione e'
    stata mancata. Escluderlo dal denominatore farebbe sparire l'errore:
    su una matrice plausibile con 900 px HSIL -> GLASS la sensibilita'
    risultava 0,854 invece di 0,700, cioe' 15 punti di troppo.

    Definizioni (H = HSIL, T = {NEGATIVE, LSIL, HSIL}, NH = T \\ {H}):
        TP = cm[H, H]
        FN = somma di tutta la riga H  -  TP        (ogni colonna, vetro incluso)
        FP = somma della colonna H sulle righe NH
        TN = somma di tutte le righe NH  -  FP
    """
    cm = cm_last.double()
    tessuto = [i for i in range(4) if i != C.IDX_GLASS_LAST]
    non_hsil = [i for i in tessuto if i != C.IDX_HSIL_LAST]

    tp = float(cm[C.IDX_HSIL_LAST, C.IDX_HSIL_LAST])
    fn = float(cm[C.IDX_HSIL_LAST, :].sum()) - tp        # tutte le colonne
    fp = float(cm[non_hsil, C.IDX_HSIL_LAST].sum())
    tn = float(cm[non_hsil, :].sum()) - fp

    # quanti pixel di tessuto finiscono erroneamente in "vetro":
    # se questo numero e' alto, c'e' un problema a monte da indagare
    px_verso_vetro = float(cm[tessuto, C.IDX_GLASS_LAST].sum())

    # NaN, non 0, quando la classe e' ASSENTE dalla verita'.
    # Con 0 una WSI o un fold senza HSIL contribuirebbe "sensibilita' 0"
    # alla media, indistinguibile da un fallimento reale. La distinzione
    # e' la stessa che macro_presenti() applica alle IoU.
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
    """
    Confine LSIL/HSIL isolato dal resto, sui soli pixel realmente
    lesionali.

    Va riportato SEPARATAMENTE e con estrema cautela: se CIN1 e HSIL
    coesistono in pochissime WSI, questo numero non e' una stima di
    popolazione. Con co-occorrenza 1/40 il modello puo' distinguerle
    riconoscendo il vetrino anziche' la morfologia.
    """
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


# --------------------------------------------------------------------
# STAMPA
# --------------------------------------------------------------------
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
    """
    Tabella per classe con il NUMERO DI PIXEL VERI accanto.

    Il conteggio non e' decorativo: una recall del 95% su 2.000 pixel e
    una su 40 milioni sono due affermazioni completamente diverse. Con
    Stroma concentrato in 2 WSI su 40, senza quella colonna si
    rischierebbe di leggere come solido un numero che poggia su
    pochissimi dati.

    Le classi assenti dalla ground truth mostrano 'n/a', non 0: e' la
    stessa distinzione che macro_presenti() fa nel calcolo.
    """
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
    """
    Classifica gli errori fuori diagonale, dal piu' grave al meno.

    Restituisce, per ogni coppia (vera, predetta), due misure:
      - quota  : frazione dei pixel di quella classe vera che finisce
                 nella classe sbagliata. Dice QUANTO E' SISTEMATICO
                 l'errore per quella classe.
      - pixel  : numero assoluto. Dice QUANTO PESA sul totale.

    Le due non coincidono e servono entrambe: un errore al 90% su una
    classe rara e' sistematico ma pesa poco; un errore al 15% su una
    classe enorme pesa molto ma potrebbe essere solo imprecisione di
    confine.

    L'ordinamento e' per quota, perche' e' quella che indica un
    problema di apprendimento e non di dimensione della classe.
    """
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