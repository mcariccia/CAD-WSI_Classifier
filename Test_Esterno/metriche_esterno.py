"""
metriche_esterno.py
====================================================================
Metriche della validazione esterna, in spazio NEGATIVE / LSIL / HSIL.

COSA SI RIPORTA E COSA NO
--------------------------
Sensibilita' e specificita' NON dipendono dalla prevalenza: sono le
uniche metriche confrontabili fra test interno e MTCHI, che hanno
composizione completamente diversa. Precisione, Dice e IoU dipendono
dalla prevalenza e vengono riportate solo accanto alla composizione,
mai come confronto fra i due domini.

Attenzione a una asimmetria che va dichiarata in tesi: MTCHI annota
SOLO l'epitelio, quindi la specificita' misurata qui e' calcolata contro
epitelio Normal. Quella interna include stroma e ghiandole, che sono
negativi facili e abbondanti (lo stroma da solo e' il 44,7% dei pixel).
Sono due numeri con lo stesso nome che misurano cose diverse, e quello
di MTCHI e' sistematicamente piu' difficile.

LE DUE POLITICHE SUL VETRO
---------------------------
Il modello privato ha una classe Sfondo/GLASS che in MTCHI non esiste.

  "negativo" (PRIMARIA): un pixel di epitelio predetto vetro non e'
      stato chiamato HSIL, quindi rispetto all'endpoint primario e' un
      negativo e pesa come tale.
  "escluso"  (di confronto): quei pixel escono dal calcolo. Gonfia le
      metriche perche' toglie dalla valutazione proprio i pixel su cui
      il modello ha fallito. Si calcola per rendere la differenza
      visibile invece che nascosta, mai da riportare da sola.

E' lo stesso schema di test.py, che riporta specificita' e specificita'
su tessuto insieme alla quota scartata.

BOOTSTRAP: PER PAZIENTE
------------------------
L'unita' statistica e' il paziente. MTCHI ha piu' RoI per alcuni
pazienti (stesso vetrino, stesso taglio, stessa colorazione):
ricampionare le RoI le tratterebbe come indipendenti e produrrebbe
intervalli assurdamente stretti. Per le metriche la cui stima puntuale
e' una MEDIA PER RoI si mediano i valori, per quelle aggregate si
sommano le matrici di confusione: sono due stimatori diversi e vanno
ricampionati coerentemente.

EQUIVALENZA CON LA METRICA UFFICIALE
-------------------------------------
La mIoU a 3 classi calcolata dalla matrice aggregata E' esattamente la
NLH-mIoU di eval_task2.py (MCPRL). Lo script lo verifica numericamente a
ogni esecuzione: se le due divergono c'e' un errore nella conversione o
nell'aggregazione, e i numeri non vanno usati.
====================================================================
"""

import numpy as np

import config_esterno as CE


# ====================================================================
# 1. MAPPATURA E MATRICE DI CONFUSIONE
# ====================================================================
_LUT_TRUTH = np.full(256, CE.IGNORE, dtype=np.uint8)
for _i, _v in enumerate(CE.LUT_TRUTH):
    _LUT_TRUTH[_i] = _v

_LUT_PRED = np.full(256, CE.IGNORE, dtype=np.uint8)
for _i, _v in enumerate(CE.LUT_PRED):
    _LUT_PRED[_i] = _v


def mappa_truth(msk_mtchi):
    """0=BG -> IGNORE, 1=Normal -> NEGATIVE, 2=CIN1 -> LSIL, 3/4 -> HSIL."""
    return _LUT_TRUTH[msk_mtchi]


def mappa_predizione(pred6, politica=None):
    """
    6 classi fini -> spazio di confronto.
    -> (pred3, maschera_vetro)
    """
    politica = politica or CE.POLITICA_VETRO
    p = _LUT_PRED[pred6]
    vetro = p == CE.SEGNAPOSTO_VETRO
    if politica == "negativo":
        p = np.where(vetro, np.uint8(CE.IDX_NEGATIVE), p)
    elif politica == "escluso":
        p = np.where(vetro, np.uint8(CE.IGNORE), p)
    else:
        raise ValueError(f"politica vetro sconosciuta: {politica}")
    return p, vetro


def confusione(pred3, truth3):
    """
    Somma i soli pixel validi su ENTRAMBI. Equivale al background_filter
    di eval_task2.py: dove la ground truth e' Background non si conta
    niente, qualunque cosa il modello abbia predetto.
    """
    n = CE.NUM_CLASSI
    cm = np.zeros((n, n), dtype=np.int64)
    valido = (truth3 != CE.IGNORE) & (pred3 != CE.IGNORE)
    if not valido.any():
        return cm
    idx = truth3[valido].astype(np.int64) * n + pred3[valido].astype(np.int64)
    cm += np.bincount(idx, minlength=n * n).reshape(n, n)
    return cm


# ====================================================================
# 2. METRICHE DA MATRICE
# ====================================================================
def iou_per_classe(cm):
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - inter
    return np.where(union > 0, inter / np.maximum(union, 1), np.nan)


def dice_per_classe(cm):
    inter = np.diag(cm).astype(np.float64)
    den = cm.sum(0) + cm.sum(1)
    return np.where(den > 0, 2 * inter / np.maximum(den, 1), np.nan)


def recall_per_classe(cm):
    s = cm.sum(1)
    return np.where(s > 0, np.diag(cm) / np.maximum(s, 1), np.nan)


def precision_per_classe(cm):
    s = cm.sum(0)
    return np.where(s > 0, np.diag(cm) / np.maximum(s, 1), np.nan)


def macro_presenti(cm):
    """
    Media della IoU sulle sole classi con supporto nella ground truth.

    Su una RoI mono-grado, mediare su tutte e tre impone un tetto di
    1/3: basta un pixel predetto in una classe assente perche' quella
    classe passi da NaN (unione zero) a 0. Una RoI segmentata al 97%
    otterrebbe 0,32. Non e' indulgenza: il falso positivo resta contato
    una volta, sulla classe vera.
    """
    idx = [k for k in range(CE.NUM_CLASSI) if cm[k].sum() > 0]
    if not idx:
        return float("nan")
    v = iou_per_classe(cm)[idx]
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else float("nan")


def nc_miou(cm):
    """NC-mIoU: Normal contro lesione. Equivale a normal_cin_IU()."""
    b = np.zeros((2, 2), dtype=np.float64)
    b[0, 0] = cm[0, 0]
    b[0, 1] = cm[0, 1] + cm[0, 2]
    b[1, 0] = cm[1, 0] + cm[2, 0]
    b[1, 1] = cm[1, 1] + cm[1, 2] + cm[2, 1] + cm[2, 2]
    v = iou_per_classe(b)
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else float("nan")


def binario_hsil(cm):
    """
    HSIL contro non-HSIL: la soglia che decide la gestione clinica
    (ASCCP 2019, escissione contro sorveglianza). E' l'endpoint primario,
    identico a quello del test interno.
    """
    h = CE.IDX_HSIL
    tp = float(cm[h, h])
    fn = float(cm[h].sum() - tp)
    fp = float(cm[:, h].sum() - tp)
    tn = float(cm.sum() - tp - fn - fp)
    return {
        "sensibilita": tp / max(tp + fn, 1),
        "specificita": tn / max(tn + fp, 1),
        "precisione": tp / max(tp + fp, 1),
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


def accuratezza(cm):
    return float(np.trace(cm) / max(cm.sum(), 1))


def composizione(cm):
    tot = max(cm.sum(), 1)
    return {CE.NOMI_CLASSI[k]: float(cm[k].sum()) / tot
            for k in range(CE.NUM_CLASSI)}


def riassunto(cm, quota_vetro=None, cm_escluso=None):
    """Blocco unico di metriche, quello che finisce nel report e in tabella."""
    b = binario_hsil(cm)
    r = {
        "miou_3classi": macro_presenti(cm),
        "nc_miou": nc_miou(cm),
        "accuratezza": accuratezza(cm),
        "iou_per_classe": [None if np.isnan(x) else float(x)
                           for x in iou_per_classe(cm)],
        "dice_per_classe": [None if np.isnan(x) else float(x)
                            for x in dice_per_classe(cm)],
        "recall_per_classe": [None if np.isnan(x) else float(x)
                              for x in recall_per_classe(cm)],
        "precision_per_classe": [None if np.isnan(x) else float(x)
                                 for x in precision_per_classe(cm)],
        "hsil_vs_non_hsil": {k: float(v) for k, v in b.items()},
        "composizione": composizione(cm),
        "pixel_valutati": int(cm.sum()),
    }
    if quota_vetro is not None:
        r["quota_predetta_vetro"] = float(quota_vetro)
    if cm_escluso is not None:
        be = binario_hsil(cm_escluso)
        r["hsil_escludendo_vetro"] = {
            "sensibilita": float(be["sensibilita"]),
            "specificita": float(be["specificita"]),
            "pixel_valutati": int(cm_escluso.sum()),
        }
        # Analogo della "specificita' su tessuto" del test interno: un
        # pixel predetto vetro non e' stato chiamato HSIL e finirebbe fra
        # i veri negativi, gonfiando la specificita'.
        r["specificita_su_tessuto"] = float(be["specificita"])
    if cm[CE.IDX_LSIL].sum() > 0:
        s = cm[CE.IDX_LSIL].sum()
        r["destino_lsil"] = {
            "verso_HSIL": float(cm[CE.IDX_LSIL, CE.IDX_HSIL] / s),
            "verso_NEGATIVE": float(cm[CE.IDX_LSIL, CE.IDX_NEGATIVE] / s),
        }
    return r


# ====================================================================
# 3. RILEVAZIONE A LIVELLO DI TILE
# ====================================================================
def rilevazione_per_tile(tile, soglie=None, frazione_minima=None):
    """
    Rilevazione a livello di TILE, identica nello spirito a test.py.

    Un tile entra nel conteggio della classe c se almeno FRAZIONE_MINIMA_LESIONE
    dei suoi pixel appartiene a c. E' "rilevato" se almeno la soglia indicata
    dei suoi pixel di c e' predetta POSITIVA (LSIL o HSIL).

    La recall per pixel sottostima la capacita' di segnalare una lesione: la
    domanda clinica non e' quanti pixel, e' se la lesione viene portata
    all'attenzione del patologo. I tassi si mediano PER PAZIENTE, non per tile:
    i tile di uno stesso vetrino sono correlati quanto i pixel.

    tile: lista di dict con 'paziente', 'n_vera' (array per classe),
          'n_positivo' (array), 'n_esatto' (array).
    """
    soglie = soglie or CE.SOGLIE_RILEVAZIONE
    fmin = CE.FRAZIONE_MINIMA_LESIONE if frazione_minima is None else frazione_minima
    n_px = CE.TILE * CE.TILE
    out = {}
    for c in (CE.IDX_LSIL, CE.IDX_HSIL):
        voci = [t for t in tile if t["n_vera"][c] >= fmin * n_px]
        if not voci:
            out[CE.NOMI_CLASSI[c]] = None
            continue
        per_paz = {}
        for t in voci:
            per_paz.setdefault(t["paziente"], []).append(t)
        blocco = {"n_tile": len(voci), "n_pazienti": len(per_paz),
                  "frazione_minima": float(fmin),
                  "positivo": {}, "grado_esatto": {}}
        for s in soglie:
            for campo, chiave in (("n_positivo", "positivo"),
                                  ("n_esatto", "grado_esatto")):
                tassi = []
                for _, g in per_paz.items():
                    tassi.append(float(np.mean(
                        [1.0 if (t[campo][c] / max(t["n_vera"][c], 1)) >= s
                         else 0.0 for t in g])))
                blocco[chiave][f"{s:.2f}"] = float(np.mean(tassi))
        out[CE.NOMI_CLASSI[c]] = blocco
    return out


# ====================================================================
# 4. BOOTSTRAP E JACKKNIFE PER PAZIENTE
# ====================================================================
def _pazienti(per_roi):
    d = {}
    for nome, v in per_roi.items():
        d.setdefault(v["paziente"], []).append(nome)
    return d


def bootstrap_aggregato(per_roi, funzione, n=None, seed=None, campo="cm"):
    """
    Ricampiona i PAZIENTI, somma le matrici di confusione di tutte le
    loro RoI, applica la funzione. Per metriche la cui stima puntuale
    viene dalla matrice aggregata.
    """
    n = n or CE.N_BOOTSTRAP
    seed = CE.SEED if seed is None else seed
    gruppi = _pazienti(per_roi)
    chiavi = sorted(gruppi)
    cm_paz = {}
    for p, nomi in gruppi.items():
        cm_paz[p] = np.sum([np.array(per_roi[x][campo]) for x in nomi], axis=0)

    rng = np.random.default_rng(seed)
    valori = []
    for _ in range(n):
        scelti = rng.choice(len(chiavi), len(chiavi), replace=True)
        cm = np.sum([cm_paz[chiavi[i]] for i in scelti], axis=0)
        v = funzione(cm)
        if v is not None and np.isfinite(v):
            valori.append(v)
    if not valori:
        return (float("nan"), float("nan"))
    return (float(np.percentile(valori, 2.5)),
            float(np.percentile(valori, 97.5)))


def bootstrap_media(per_roi, campo, n=None, seed=None):
    """
    Ricampiona i pazienti e MEDIA i valori per RoI. Per metriche la cui
    stima puntuale e' gia' una media per RoI: usare la somma delle
    matrici sarebbe uno stimatore diverso da quello riportato.
    """
    n = n or CE.N_BOOTSTRAP
    seed = CE.SEED if seed is None else seed
    gruppi = _pazienti(per_roi)
    chiavi = sorted(gruppi)
    val_paz = {}
    for p, nomi in gruppi.items():
        v = [per_roi[x][campo] for x in nomi
             if per_roi[x][campo] is not None and np.isfinite(per_roi[x][campo])]
        val_paz[p] = float(np.mean(v)) if v else np.nan

    rng = np.random.default_rng(seed)
    valori = []
    for _ in range(n):
        scelti = rng.choice(len(chiavi), len(chiavi), replace=True)
        v = [val_paz[chiavi[i]] for i in scelti]
        v = [x for x in v if np.isfinite(x)]
        if v:
            valori.append(float(np.mean(v)))
    if not valori:
        return (float("nan"), float("nan"))
    return (float(np.percentile(valori, 2.5)),
            float(np.percentile(valori, 97.5)))


def jackknife_paziente(per_roi, funzione, campo="cm"):
    """
    Lascia fuori un paziente per volta. Dice se la stima poggia su un
    solo soggetto: con pochi pazienti e' informativo quanto il bootstrap
    e molto piu' leggibile.
    """
    gruppi = _pazienti(per_roi)
    cm_paz = {p: np.sum([np.array(per_roi[x][campo]) for x in nomi], axis=0)
              for p, nomi in gruppi.items()}
    out = {}
    for escluso in sorted(cm_paz):
        cm = np.sum([v for p, v in cm_paz.items() if p != escluso], axis=0)
        val = funzione(cm)
        out[escluso] = None if val is None or not np.isfinite(val) else float(val)
    return out


# ====================================================================
# 5. CONTROLLO DI EQUIVALENZA CON eval_task2.py
# ====================================================================
def nlh_miou_ufficiale(coppie):
    """
    Riproduce alla lettera normal_lsil_hsil_IU() + l'aggregazione di
    evaluation() (MCPRL 2019) sugli array gia' rimappati. Controllo
    indipendente: se non coincide con la mIoU da matrice di confusione,
    c'e' un errore da qualche parte e i numeri non vanno usati.
    """
    inter = np.zeros(CE.NUM_CLASSI)
    union = np.zeros(CE.NUM_CLASSI)
    for pred, truth in coppie:
        valido = truth != CE.IGNORE
        p = np.where(valido, pred, 255).astype(np.int16)
        t = np.where(valido, truth, 255).astype(np.int16)
        for k in range(CE.NUM_CLASSI):
            sp, st = (p == k), (t == k)
            inter[k] += np.count_nonzero(sp & st)
            union[k] += np.count_nonzero(sp | st)
    return float(np.sum(inter / np.maximum(union, 1)) / CE.NUM_CLASSI)