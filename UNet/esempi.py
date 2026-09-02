"""
esempi.py
====================================================================
Esempi qualitativi per la tesi: tile, ground truth, predizione e mappa
degli errori, salvati in una cartella 'esempi'.

QUALE MODELLO
--------------
L'ENSEMBLE con decisione ARGMAX, cioe' esattamente il sistema di cui si
riportano le metriche principali. Non "il fold migliore":

  - i punteggi dei fold sono punteggi di VALIDATION, e sceglierne uno
    perche' ha fatto il numero piu' alto e' selezione sui dati di
    validazione, la stessa che si evita ovunque altrove;
  - le figure qualitative devono illustrare il sistema misurato. Un
    overlay prodotto da un modello diverso da quello del risultato
    principale e' un disallineamento silenzioso fra testo e figure;
  - sul test set i modelli singoli sono comunque peggiori dell'ensemble.

L'ensemble e' lecito qui come lo e' per le metriche: nessuno dei modelli
dei fold ha mai visto le WSI di test.

COME SI SCELGONO LE TILE
-------------------------
La regola e' PRESPECIFICATA e deterministica. Nessuna tile viene scelta
guardandola: si scelgono i percentili di una distribuzione.

  1. Per ogni classe si considerano solo le tile in cui la classe copre
     almeno FRAZIONE_MINIMA dei pixel ANNOTATI. Sotto quella soglia la
     IoU per tile e' rumore e l'esempio non e' rappresentativo di nulla.
  2. Le tile ammesse si ordinano per IoU di quella classe e si prendono
     i percentili 10, 50 e 90: caso difficile, caso tipico, caso
     riuscito. Mai il massimo.
  3. Al massimo UNA tile per WSI per classe. Senza questo vincolo tutti
     gli esempi di HSIL verrebbero da un solo vetrino (nel test attuale
     Prova 5 porta il 68,5% dell'HSIL) e tutti quelli di CIN1 da un
     altro (Prova 10, 66,5%): la figura mostrerebbe un vetrino, non un
     modello.
  4. Si aggiungono le tile MEDIANE dei principali errori fuori
     diagonale della matrice aggregata. Sono le figure che spiegano
     qualcosa invece di limitarsi a mostrare un buon risultato.

Tutte le scelte finiscono in manifest.json con percentile, IoU e
conteggi, cosi' la figura e' riproducibile e la regola e' citabile in
didascalia.

ZONE NON ANNOTATE
------------------
Meta' delle tile e' mono-classe con ampie aree non annotate. Quelle aree
NON sono errori: non c'e' una verita' con cui confrontarsi. Vengono
disegnate in grigio scuro e non entrano in nessun conteggio. Colorarle
di rosso sarebbe una falsificazione, colorarle di verde anche.
====================================================================
"""

import json
import os

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import config as C
import dati as D

# ====================================================================
# PARAMETRI DELLA REGOLA DI SELEZIONE
# ====================================================================
NOME_CARTELLA = "esempi"

# Una classe entra fra i candidati solo se copre almeno questa frazione
# dei pixel annotati della tile.
FRAZIONE_MINIMA = 0.05

# Percentili estratti, con l'etichetta che finisce nel titolo.
PERCENTILI = ((10, "difficile"), (50, "tipico"), (90, "riuscito"))

# Tile per WSI per classe. 1 = massima diversita' di vetrini.
MAX_PER_WSI = 1

# Errori fuori diagonale da illustrare, presi dalla matrice aggregata.
N_ERRORI = 3
# Una tile e' candidata per l'errore (i -> j) se almeno questa frazione
# dei suoi pixel di classe i finisce in j.
FRAZIONE_MINIMA_ERRORE = 0.15

# Tile su cui confrontare argmax e costo minimo.
N_CONFRONTI_DECISIONE = 2

ALPHA_OVERLAY = 0.55

# Palette delle classi fini. Fissa: la stessa in ogni figura della tesi.
COLORI_CLASSI = np.array([
    [245, 245, 245],   # Sfondo    bianco sporco
    [255, 160,  30],   # CIN1      arancione
    [ 55, 135, 215],   # Ghiandole blu
    [200,  35,  50],   # HSIL      rosso
    [ 70, 175,  95],   # Mucosa    verde
    [180, 120, 200],   # Stroma    viola
], dtype=np.uint8)
COLORE_NON_ANNOTATO = np.array([70, 70, 70], dtype=np.uint8)

# Mappa degli errori: palette diversa da quella delle classi, per non
# indurre a leggere il verde come "Mucosa".
COLORE_CORRETTO = np.array([ 60, 170, 100], dtype=np.uint8)
COLORE_ERRORE = np.array([225,  25,  90], dtype=np.uint8)


def cartella():
    d = os.path.join(C.RISULTATI_DIR, NOME_CARTELLA)
    os.makedirs(d, exist_ok=True)
    return d


# ====================================================================
# RACCOLTA DELLE STATISTICHE PER TILE
# ====================================================================
class RaccoglitoreTile:
    """
    Durante il passaggio principale sul test set registra, per ogni
    tile, la sua matrice di confusione 6x6 sui soli pixel annotati.

    Sei per sei interi per tile sono qualche centinaio di kilobyte in
    totale: da li' si ricavano IoU per classe, conteggi di errore e
    frazioni di copertura senza dover ripetere l'inferenza per
    decidere cosa mostrare. L'inferenza si ripete solo sulle poche tile
    effettivamente selezionate.

    ASSUNZIONE: il DataLoader e' costruito con shuffle=False e senza
    sampler, quindi la posizione nel loader corrisponde all'indice in
    'campioni'. Se un giorno cambia, l'assert in registra() lo dice
    subito invece di produrre figure con la tile sbagliata.
    """

    def __init__(self, campioni):
        self.campioni = campioni
        n = len(campioni)
        self.cm = np.zeros((n, C.NUM_CLASSES, C.NUM_CLASSES), dtype=np.int64)
        self.visti = np.zeros(n, dtype=bool)

    @torch.no_grad()
    def registra(self, base, pred, masks, wsis):
        B = pred.shape[0]
        assert base + B <= len(self.campioni), (
            "Piu' tile del previsto: il loader non e' in ordine sequenziale.")
        n = C.NUM_CLASSES
        valido = masks < n
        p = pred.reshape(B, -1)
        t = masks.reshape(B, -1)
        v = valido.reshape(B, -1)
        for b in range(B):
            i = base + b
            atteso = os.path.basename(self.campioni[i][0])
            assert self.campioni[i][2] == wsis[b], (
                f"disallineamento loader/campioni alla tile {i}: "
                f"attesa {self.campioni[i][2]}, ricevuta {wsis[b]} "
                f"({atteso})")
            m = v[b]
            if not bool(m.any()):
                self.visti[i] = True
                continue
            idx = t[b][m] * n + p[b][m]
            cm = torch.bincount(idx, minlength=n * n).reshape(n, n)
            self.cm[i] = cm.cpu().numpy()
            self.visti[i] = True

    # ----------------------------------------------------------------
    def iou_classe(self, c):
        """IoU per tile della classe c, NaN dove l'unione e' vuota."""
        veri = self.cm[:, c, :].sum(1)
        predetti = self.cm[:, :, c].sum(1)
        inter = self.cm[:, c, c]
        unione = veri + predetti - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(unione > 0, inter / unione, np.nan)

    def copertura(self, c):
        tot = self.cm.sum((1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(tot > 0, self.cm[:, c, :].sum(1) / tot, 0.0)

    def quota_errore(self, i, j):
        veri = self.cm[:, i, :].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(veri > 0, self.cm[:, i, j] / veri, 0.0)


# ====================================================================
# SELEZIONE
# ====================================================================
def _scegli_percentili(valori, ammessi, wsi, max_per_wsi=MAX_PER_WSI):
    """
    Percentili per rango sulle tile ammesse, con al piu' max_per_wsi
    tile per vetrino.

    Si scorre dal percentile richiesto verso l'alto e verso il basso
    finche' non si trova una tile di un vetrino non ancora usato: cosi'
    il vincolo di diversita' sposta la scelta il meno possibile, invece
    di scartare il percentile.
    """
    idx = [i for i in ammessi if not np.isnan(valori[i])]
    if not idx:
        return []
    idx.sort(key=lambda i: valori[i])
    usati, scelte = {}, []
    for perc, etichetta in PERCENTILI:
        posizione = int(round((perc / 100.0) * (len(idx) - 1)))
        trovato = None
        for salto in range(len(idx)):
            for k in (posizione - salto, posizione + salto):
                if not 0 <= k < len(idx):
                    continue
                i = idx[k]
                if usati.get(wsi[i], 0) < max_per_wsi and i not in scelte:
                    trovato = (i, k)
                    break
            if trovato:
                break
        if trovato is None:
            continue
        i, k = trovato
        usati[wsi[i]] = usati.get(wsi[i], 0) + 1
        scelte.append(i)
        yield {"indice": int(i), "percentile": perc, "etichetta": etichetta,
               "rango": k + 1, "n_candidati": len(idx),
               "valore": float(valori[i])}


def seleziona(racc, cm_aggregata):
    """
    -> lista di voci {indice, tipo, classe, etichetta, valore, ...}

    Prima i percentili per classe, poi le tile mediane dei principali
    errori fuori diagonale.
    """
    wsi = [c[2] for c in racc.campioni]
    voci = []

    for c in range(C.NUM_CLASSES):
        if c == C.CLASSE_SFONDO:
            continue
        cop = racc.copertura(c)
        ammessi = np.where(cop >= FRAZIONE_MINIMA)[0].tolist()
        if not ammessi:
            continue
        iou = racc.iou_classe(c)
        for v in _scegli_percentili(iou, ammessi, wsi):
            v.update({"tipo": "classe", "classe": C.CLASS_NAMES[c],
                      "indice_classe": c, "metrica": "IoU per tile",
                      "copertura": float(cop[v["indice"]])})
            voci.append(v)

    # ---- errori dominanti ----
    cm = np.asarray(cm_aggregata, dtype=np.float64)
    righe = cm.sum(1)
    coppie = []
    for i in range(C.NUM_CLASSES):
        if righe[i] == 0 or i == C.CLASSE_SFONDO:
            continue
        for j in range(C.NUM_CLASSES):
            if i != j:
                coppie.append((cm[i, j] / righe[i], i, j))
    coppie.sort(reverse=True)

    for quota, i, j in coppie[:N_ERRORI]:
        q = racc.quota_errore(i, j)
        cop = racc.copertura(i)
        ammessi = np.where((cop >= FRAZIONE_MINIMA) &
                           (q >= FRAZIONE_MINIMA_ERRORE))[0].tolist()
        if not ammessi:
            continue
        ammessi.sort(key=lambda k: q[k])
        scelto = ammessi[len(ammessi) // 2]        # mediana, non il massimo
        voci.append({
            "indice": int(scelto), "tipo": "errore",
            "classe": C.CLASS_NAMES[i], "indice_classe": i,
            "predetta": C.CLASS_NAMES[j], "indice_predetta": j,
            "etichetta": "mediana dell'errore",
            "metrica": f"quota {C.CLASS_NAMES[i]} -> {C.CLASS_NAMES[j]}",
            "valore": float(q[scelto]),
            "quota_aggregata": float(quota),
            "n_candidati": len(ammessi),
            "copertura": float(cop[scelto]),
        })
    return voci


# ====================================================================
# RENDER
# ====================================================================
def _colora_maschera(mask):
    out = np.empty((*mask.shape, 3), dtype=np.uint8)
    out[:] = COLORE_NON_ANNOTATO
    valido = mask < C.NUM_CLASSES
    out[valido] = COLORI_CLASSI[mask[valido]]
    return out


def _mappa_errori(mask, pred):
    out = np.empty((*mask.shape, 3), dtype=np.uint8)
    out[:] = COLORE_NON_ANNOTATO
    valido = mask < C.NUM_CLASSES
    corretto = valido & (pred == mask)
    out[corretto] = COLORE_CORRETTO
    out[valido & ~corretto] = COLORE_ERRORE
    return out


def _sovrapponi(rgb, colori, alpha=ALPHA_OVERLAY):
    return (rgb.astype(np.float32) * (1 - alpha) +
            colori.astype(np.float32) * alpha).astype(np.uint8)


@torch.no_grad()
def _predici_tile(modelli, p_img, device, amp, softmax_fn, omega=None):
    """Rilegge la tile da disco e rifa' l'inferenza con lo stesso TTA."""
    bgr = cv2.imread(p_img)
    if bgr is None:
        raise RuntimeError(f"immagine illeggibile: {p_img}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = D.transforms_val()(image=rgb)["image"].unsqueeze(0).to(device)
    prob = softmax_fn(modelli, x, amp)
    if omega is None:
        pred = prob.argmax(1)
    else:
        pred = torch.einsum("bihw,ij->bjhw", prob, omega).argmin(1)
    return rgb, pred[0].cpu().numpy().astype(np.int64), prob[0].cpu().numpy()


def _legenda(fig):
    voci = [Patch(facecolor=COLORI_CLASSI[i] / 255, edgecolor="black",
                  linewidth=0.4, label=C.CLASS_NAMES[i])
            for i in range(C.NUM_CLASSES)]
    voci.append(Patch(facecolor=COLORE_NON_ANNOTATO / 255, edgecolor="black",
                      linewidth=0.4, label="non annotato"))
    fig.legend(handles=voci, loc="lower center", ncol=len(voci),
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))


def _figura(rgb, mask, pred, titolo, percorso, sottotitolo=""):
    fig, ax = plt.subplots(1, 4, figsize=(16.5, 4.6))
    ax[0].imshow(rgb)
    ax[0].set_title("tile (H&E, normalizzata)", fontsize=10)
    ax[1].imshow(_sovrapponi(rgb, _colora_maschera(mask)))
    ax[1].set_title("ground truth", fontsize=10)
    ax[2].imshow(_sovrapponi(rgb, _colora_maschera(pred)))
    ax[2].set_title("predizione", fontsize=10)
    ax[3].imshow(_sovrapponi(rgb, _mappa_errori(mask, pred)))
    ax[3].set_title("verde = corretto   rosa = errore   "
                    "grigio = non annotato", fontsize=9)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(titolo, fontsize=11, weight="bold")
    if sottotitolo:
        fig.text(0.5, 0.925, sottotitolo, ha="center", fontsize=8.5,
                 color="#444444")
    _legenda(fig)
    plt.tight_layout(rect=(0, 0.03, 1, 0.90))
    plt.savefig(percorso, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _figura_decisioni(rgb, mask, pred_a, pred_c, titolo, percorso):
    """argmax contro costo minimo sulle STESSE probabilita'."""
    fig, ax = plt.subplots(1, 4, figsize=(16.5, 4.6))
    ax[0].imshow(_sovrapponi(rgb, _colora_maschera(mask)))
    ax[0].set_title("ground truth", fontsize=10)
    ax[1].imshow(_sovrapponi(rgb, _colora_maschera(pred_a)))
    ax[1].set_title("decisione argmax", fontsize=10)
    ax[2].imshow(_sovrapponi(rgb, _colora_maschera(pred_c)))
    ax[2].set_title("decisione a costo minimo", fontsize=10)
    diff = np.zeros((*mask.shape, 3), dtype=np.uint8)
    diff[:] = COLORE_NON_ANNOTATO
    uguali = pred_a == pred_c
    diff[uguali] = COLORE_CORRETTO
    diff[~uguali] = COLORE_ERRORE
    ax[3].imshow(_sovrapponi(rgb, diff))
    quota = float((~uguali).mean())
    ax[3].set_title(f"rosa = decisione diversa ({100*quota:.1f}% dei pixel)",
                    fontsize=9)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(titolo, fontsize=11, weight="bold")
    fig.text(0.5, 0.925, "Stesse probabilita', regola di scelta diversa. "
             "Omega e' prespecificata in config.", ha="center", fontsize=8.5,
             color="#444444")
    _legenda(fig)
    plt.tight_layout(rect=(0, 0.03, 1, 0.90))
    plt.savefig(percorso, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _nome_file(voce, campione):
    tile = os.path.splitext(os.path.basename(campione[0]))[0]
    wsi = campione[2].replace(" ", "")
    if voce["tipo"] == "errore":
        base = f"errore_{voce['classe']}_verso_{voce['predetta']}"
    else:
        base = f"{voce['classe']}_p{voce['percentile']:02d}_{voce['etichetta']}"
    return f"{base}__{wsi}__{tile}.png"


# ====================================================================
# ENTRY POINT
# ====================================================================
def salva_esempi(racc, cm_aggregata, modelli, device, amp, softmax_fn,
                 omega=None, etichetta_sistema=""):
    """
    Produce la cartella 'esempi' e restituisce il manifest.

    softmax_fn e' la stessa funzione usata per le metriche (in test.py
    _softmax_batch): passandola invece di reimplementarla si evita che
    le figure mostrino un TTA o una media diversi da quelli misurati.
    """
    d = cartella()
    voci = seleziona(racc, cm_aggregata)
    if not voci:
        print("   nessuna tile soddisfa la regola di selezione.")
        return None

    print(f"\n--- ESEMPI QUALITATIVI ---")
    print(f"   sistema: {etichetta_sistema or 'ensemble, argmax'}")
    print(f"   regola: percentili {', '.join(str(p) for p, _ in PERCENTILI)} "
          f"della IoU per tile, copertura minima "
          f"{100*FRAZIONE_MINIMA:.0f}%, max {MAX_PER_WSI} tile per WSI")
    print(f"   {'file':<52}{'classe':<12}{'caso':<20}{'valore':>8}")
    print("   " + "-" * 92)

    manifest = {
        "sistema": etichetta_sistema or "ensemble, decisione argmax",
        "n_modelli": len(modelli),
        "regola": {
            "frazione_minima_copertura": FRAZIONE_MINIMA,
            "percentili": [p for p, _ in PERCENTILI],
            "max_tile_per_wsi": MAX_PER_WSI,
            "n_errori_illustrati": N_ERRORI,
            "frazione_minima_errore": FRAZIONE_MINIMA_ERRORE,
            "nota": ("Selezione deterministica sui percentili della IoU per "
                     "tile. Nessuna tile e' stata scelta osservandola."),
        },
        "esempi": [],
    }

    for voce in voci:
        camp = racc.campioni[voce["indice"]]
        rgb, pred, _ = _predici_tile(modelli, camp[0], device, amp,
                                     softmax_fn, omega=None)
        mask = D.leggi_maschera(camp[1])

        if voce["tipo"] == "errore":
            titolo = (f"{camp[2]} — errore {voce['classe']} -> "
                      f"{voce['predetta']} — tile mediana "
                      f"({100*voce['valore']:.0f}% dei pixel "
                      f"{voce['classe']} della tile)")
            caso = f"{voce['classe']}->{voce['predetta']}"
        else:
            titolo = (f"{camp[2]} — {voce['classe']} — caso "
                      f"{voce['etichetta']} (percentile {voce['percentile']}) "
                      f"— IoU tile {voce['valore']:.3f}")
            caso = f"p{voce['percentile']} {voce['etichetta']}"

        sotto = (f"{etichetta_sistema or 'ensemble, argmax'}   |   "
                 f"rango {voce.get('rango', '-')} su {voce['n_candidati']} "
                 f"tile candidate   |   copertura della classe "
                 f"{100*voce['copertura']:.0f}%")
        nome = _nome_file(voce, camp)
        _figura(rgb, mask, pred, titolo, os.path.join(d, nome), sotto)

        print(f"   {nome[:50]:<52}{voce['classe']:<12}{caso:<20}"
              f"{voce['valore']:>8.3f}")
        manifest["esempi"].append({
            **voce, "file": nome, "wsi": camp[2],
            "tile": os.path.basename(camp[0]),
            "path_immagine": camp[0], "path_maschera": camp[1],
        })

    # ---- confronto fra regole di decisione ----
    if omega is not None and N_CONFRONTI_DECISIONE > 0:
        candidati = [v for v in voci if v["tipo"] == "errore"]
        candidati += [v for v in voci if v["tipo"] == "classe"
                      and v["percentile"] == 50]
        for voce in candidati[:N_CONFRONTI_DECISIONE]:
            camp = racc.campioni[voce["indice"]]
            rgb, pred_a, _ = _predici_tile(modelli, camp[0], device, amp,
                                           softmax_fn, omega=None)
            _, pred_c, _ = _predici_tile(modelli, camp[0], device, amp,
                                         softmax_fn, omega=omega)
            mask = D.leggi_maschera(camp[1])
            nome = f"decisioni__{camp[2].replace(' ', '')}__" \
                   f"{os.path.splitext(os.path.basename(camp[0]))[0]}.png"
            _figura_decisioni(rgb, mask, pred_a, pred_c,
                              f"{camp[2]} — argmax contro costo minimo",
                              os.path.join(d, nome))
            print(f"   {nome[:50]:<52}{'—':<12}{'confronto decisioni':<20}")
            manifest.setdefault("confronti_decisione", []).append(
                {"file": nome, "wsi": camp[2],
                 "tile": os.path.basename(camp[0])})

    path_manifest = os.path.join(d, "manifest.json")
    with open(path_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n   {len(manifest['esempi'])} esempi in {d}")
    print(f"   manifest: {path_manifest}")
    print("   La regola di selezione e' nel manifest: citarla in didascalia")
    print("   e' cio' che distingue un esempio da una tile scelta a mano.")
    return manifest