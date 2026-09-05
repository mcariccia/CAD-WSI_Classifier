"""
grafici_esterno.py
====================================================================
Tabelle, DUE grafici, esempi qualitativi ed export.

PERCHE' SOLO DUE GRAFICI
-------------------------
  confusione.png                 (una per configurazione)
      Matrice 3x3 normalizzata per riga e 2x2 HSIL contro non-HSIL.
      Dice DOVE sbaglia, non solo quanto.

  confronto_configurazioni.png   (una sola, globale)
      Le quattro celle su sensibilita' HSIL, specificita' su tessuto e
      mIoU, con IC 95% bootstrap per paziente. E' LA figura
      dell'esperimento: mostra il compromesso fra prestazione
      in-dominio e robustezza cross-dominio.

Tutto il resto — per RoI, per paziente, per partizione, per fold — sta
nel report.json e nelle tabelle stampate a terminale. Un grafico che
ripete un numero gia' in tabella non aggiunge informazione, aggiunge
pagine.

Le barre di errore sono INTERVALLI DI CONFIDENZA, non deviazioni
standard: si riporta l'incertezza sulla stima, non la dispersione fra
RoI, che e' un'altra quantita' e verrebbe letta male.

ESEMPI: STESSE CONVENZIONI DI esempi.py DEL PRIVATO
----------------------------------------------------
Quattro pannelli, stessa disposizione, stessa palette, stessa regola di
selezione, stessi nomi di file. Le figure delle due sezioni della tesi
devono leggersi come una serie sola.

  1. per classe: percentili 10 / 50 / 90 di IoU -> difficile, tipico,
     riuscito. Mai il massimo. Al massimo uno per RoI per classe.
  2. per errore: le confusioni piu' frequenti nella matrice aggregata.
  3. per decisione: argmax contro costo minimo, sui tile dove
     differiscono di piu'.

I pixel non annotati (Background di MTCHI: nero fuori RoI, vetro,
stroma, ghiandole) sono grigi e non entrano in nessun conteggio.
Colorarli di rosso o di verde sarebbe una falsificazione.
====================================================================
"""

import json
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import config_esterno as CM
import metriche_esterno as ME

COLORI_CONFIG = {"vahadane_no_k2": "#9e9e9e", "vahadane_si_k2": "#1a73e8",
                 "vahadane_no_k3": "#f9ab00", "vahadane_si_k3": "#d93025"}

_PAL = np.array(CM.COLORI_CLASSI, dtype=np.uint8)
_NON_ANN = np.array(CM.COLORE_NON_ANNOTATO, dtype=np.uint8)
_VETRO = np.array(CM.COLORE_VETRO_PRED, dtype=np.uint8)
_OK = np.array(CM.COLORE_CORRETTO, dtype=np.uint8)
_ERR = np.array(CM.COLORE_ERRORE, dtype=np.uint8)


def riga(c="-", n=78):
    print(c * n)


def _f(x):
    return "-" if x is None or not np.isfinite(x) else f"{x:.4f}"


def _ic(t):
    return "-" if not np.isfinite(t[0]) else f"[{t[0]:.3f},{t[1]:.3f}]"


# ====================================================================
# 1. TABELLE
# ====================================================================
COLONNE = ["configurazione", "decisione", "n_pazienti", "n_roi", "n_tile",
           "sens_HSIL", "sens_HSIL_IC95", "spec", "spec_su_tessuto",
           "spec_su_tessuto_IC95", "recall_NEGATIVE", "recall_LSIL",
           "mIoU_3cl", "mIoU_3cl_IC95", "NC_mIoU", "quota_pred_vetro",
           "px_valutati_M"]


def riga_tabella(nome_config, decisione, report, ris, ic, q_vetro, cm):
    b = ris["hsil_vs_non_hsil"]
    return {
        "configurazione": nome_config, "decisione": decisione,
        "n_pazienti": report["n_pazienti"], "n_roi": report["n_roi"],
        "n_tile": report["n_tile"],
        "sens_HSIL": _f(b["sensibilita"]),
        "sens_HSIL_IC95": _ic(ic["sens_hsil"]),
        "spec": _f(b["specificita"]),
        "spec_su_tessuto": _f(ris.get("specificita_su_tessuto")),
        "spec_su_tessuto_IC95": _ic(ic["spec_tessuto"]),
        "recall_NEGATIVE": _f(ris["recall_per_classe"][CM.IDX_NEGATIVE]),
        "recall_LSIL": _f(ris["recall_per_classe"][CM.IDX_LSIL]),
        "mIoU_3cl": _f(ris["miou_3classi"]),
        "mIoU_3cl_IC95": _ic(ic["miou"]),
        "NC_mIoU": _f(ris["nc_miou"]),
        "quota_pred_vetro": _f(q_vetro),
        "px_valutati_M": f"{np.asarray(cm).sum()/1e6:.1f}",
    }


def stampa_tabella(righe):
    print()
    riga("=")
    print(f"  {'configurazione':<17}{'dec.':<14}{'sensHSIL':>21}"
          f"{'specTessuto':>21}{'mIoU':>21}{'vetro':>9}")
    riga("-")
    for r in righe:
        print(f"  {r['configurazione']:<17}{r['decisione']:<14}"
              f"{r['sens_HSIL']:>10} {r['sens_HSIL_IC95']:>10}"
              f"{r['spec_su_tessuto']:>10} {r['spec_su_tessuto_IC95']:>10}"
              f"{r['mIoU_3cl']:>10} {r['mIoU_3cl_IC95']:>10}"
              f"{r['quota_pred_vetro']:>9}")
    riga("=")


def salva_tabella(righe, percorso):
    with open(percorso, "w", encoding="utf-8") as f:
        f.write(";".join(COLONNE) + "\n")
        for r in righe:
            f.write(";".join(str(r.get(c, "")) for c in COLONNE) + "\n")
    return percorso


def stampa_rilevazione(ril):
    if not ril:
        return
    print("\n  RILEVAZIONE PER TILE (tassi mediati per paziente)")
    print("  Un tile e' 'rilevato' se almeno la quota indicata dei suoi pixel")
    print("  lesionali e' predetta POSITIVA (LSIL o HSIL). E' la domanda")
    print("  clinica: la lesione viene segnalata al patologo?")
    intest = "".join(f"{'>=' + str(int(100*s)) + '%':>9}"
                     for s in CM.SOGLIE_RILEVAZIONE)
    print(f"\n  {'classe':<9}{'n tile':>8}{'n paz.':>8}{intest}{'grado ok':>11}")
    riga(".")
    for nome, b in ril.items():
        if b is None:
            print(f"  {nome:<9}{'assente':>8}")
            continue
        v = "".join(f"{b['positivo'][f'{s:.2f}']:>9.3f}"
                    for s in CM.SOGLIE_RILEVAZIONE)
        g = b["grado_esatto"][f"{CM.SOGLIE_RILEVAZIONE[0]:.2f}"]
        print(f"  {nome:<9}{b['n_tile']:>8}{b['n_pazienti']:>8}{v}{g:>11.3f}")
    riga(".")


def stampa_per_fold(per_fold, ensemble):
    """
    Sensibilita' HSIL dei singoli fold contro quella dell'ensemble.

    NON serve a scegliere un fold. Il sistema riportato in tesi e'
    l'ensemble, e sceglierne uno guardando MTCHI sarebbe selezione sul test
    set. Serve a due cose:
      - dire se l'ensemble e' trainato da un solo fold o e' un consenso;
      - quantificare la varianza fra fold, che e' varianza di ADDESTRAMENTO
        e in nessun altro modo si vede.
    """
    if not per_fold:
        return
    v = [x for x in per_fold.values() if x is not None and np.isfinite(x)]
    print("\n  SENSIBILITA' HSIL PER FOLD (diagnostica, non selezione)")
    print(f"  {'fold':<12}{'sens HSIL':>12}")
    riga(".")
    for k in sorted(per_fold):
        print(f"  {k:<12}{per_fold[k]:>12.4f}")
    if v:
        print(f"  {'min - max':<12}{min(v):>12.4f} - {max(v):.4f}")
        print(f"  {'media':<12}{float(np.mean(v)):>12.4f}")
    print(f"  {'ENSEMBLE':<12}{ensemble:>12.4f}")
    riga(".")
    if v and ensemble >= max(v):
        print("  L'ensemble supera ogni singolo fold: e' un consenso, non un")
        print("  fold fortunato. E' l'argomento per riportarlo come sistema.")
    elif v and ensemble < float(np.mean(v)):
        print("  L'ensemble sta sotto la media dei fold: i fold disaccordano")
        print("  in modo sistematico, va indagato prima di interpretare il resto.")


def stampa_stratificazioni(ris):
    s = ris.get("stratificazione_copertura")
    if s:
        print(f"\n  DIPENDE DAI TILE DI BORDO?  (copertura >= "
              f"{100*s['soglia']:.0f}%)")
        print(f"  {'insieme':<24}{'sensHSIL':>11}{'specHSIL':>11}"
              f"{'mIoU':>9}{'px':>15}")
        riga(".")
        for k, et in (("tutti", "tutti i tile"),
                      ("solo_alta_copertura",
                       f">= {100*s['soglia']:.0f}% annotato")):
            v = s[k]
            print(f"  {et:<24}{v['sens_hsil']:>11.4f}{v['spec_hsil']:>11.4f}"
                  f"{v['miou']:>9.4f}{v['px']:>15,}")
        riga(".")
        print("  Se coincidono, il risultato non dipende dalla giunzione")
        print("  epitelio-stroma. Se divergono, e' una misura di quanto il")
        print("  modello soffra proprio li'.")

    p = ris.get("per_partizione")
    if p:
        print("\n  PARTIZIONI UFFICIALI DI MTCHI")
        print("  Primaria = tutte le RoI, l'unica con potenza sufficiente.")
        print("  Secondaria pre-specificata = solo il Test ufficiale, cieco")
        print("  anche rispetto alla calibrazione cromatica di K=3.0.")
        print(f"\n  {'partizione':<14}{'n RoI':>7}{'n paz.':>8}{'sensHSIL':>11}"
              f"{'specHSIL':>11}{'mIoU':>9}")
        riga(".")
        for part, v in p.items():
            print(f"  {part:<14}{v['n_roi']:>7}{v['n_pazienti']:>8}"
                  f"{v['sens_hsil']:>11.4f}{v['spec_hsil']:>11.4f}"
                  f"{v['miou_3classi']:>9.4f}")
        riga(".")


# ====================================================================
# 2. GRAFICI
# ====================================================================
def _matrice(ax, cm, nomi, titolo, cmap="Blues"):
    cm = np.asarray(cm, dtype=np.float64)
    norm = 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(len(nomi)))
    ax.set_yticks(range(len(nomi)))
    ax.set_xticklabels(nomi, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(nomi, fontsize=9)
    ax.set_xlabel("predetta", fontsize=10)
    ax.set_ylabel("vera", fontsize=10)
    ax.set_title(titolo, fontsize=11)
    for i in range(len(nomi)):
        for j in range(len(nomi)):
            ax.text(j, i, f"{norm[i, j]:.1f}", ha="center", va="center",
                    fontsize=9, color="white" if norm[i, j] > 55 else "black")


def plot_confusione(cm, bin_hsil, percorso, sottotitolo=""):
    cm2 = np.array([[bin_hsil["tn"], bin_hsil["fp"]],
                    [bin_hsil["fn"], bin_hsil["tp"]]], dtype=np.float64)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    _matrice(ax[0], cm, CM.NOMI_CLASSI, "3 classi LAST (% per riga)")
    _matrice(ax[1], cm2, ["non-HSIL", "HSIL"],
             "endpoint primario (% per riga)", cmap="Reds")
    if sottotitolo:
        fig.suptitle(sottotitolo, fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93 if sottotitolo else 1))
    fig.savefig(percorso, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return percorso


def plot_confronto(riepilogo, percorso):
    metriche = [("sens_hsil", "sensibilita HSIL"),
                ("spec_tessuto", "specificita su tessuto"),
                ("miou", "mIoU 3 classi (= NLH-mIoU ufficiale)")]
    nomi = [c["nome"] for c in CM.CONFIGURAZIONI if c["nome"] in riepilogo]
    if not nomi:
        return None
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=True)
    x = np.arange(len(nomi))
    for a, (k, tit) in zip(ax, metriche):
        val = [riepilogo[c].get(k, np.nan) for c in nomi]
        ic = [riepilogo[c].get("ic", {}).get(k, (np.nan, np.nan)) for c in nomi]
        lo = [max(v - i[0], 0) if np.isfinite(i[0]) else 0
              for v, i in zip(val, ic)]
        hi = [max(i[1] - v, 0) if np.isfinite(i[1]) else 0
              for v, i in zip(val, ic)]
        a.bar(x, val, color=[COLORI_CONFIG.get(c, "#666") for c in nomi],
              width=0.62, edgecolor="black", linewidth=0.5)
        a.errorbar(x, val, yerr=[lo, hi], fmt="none", ecolor="black",
                   elinewidth=1.1, capsize=4)
        for i, v in enumerate(val):
            if np.isfinite(v):
                a.text(i, min(v + hi[i] + 0.025, 1.0), f"{v:.3f}",
                       ha="center", fontsize=8.5)
        a.set_xticks(x)
        a.set_xticklabels([c.replace("vahadane_", "vah.") for c in nomi],
                          rotation=20, ha="right", fontsize=8.5)
        a.set_title(tit, fontsize=10)
        a.set_ylim(0, 1.05)
        a.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax[0].set_ylabel("valore (IC 95% bootstrap per paziente)", fontsize=9)
    fig.suptitle("Validazione esterna su MTCHI Task 2 — modelli congelati, "
                 "nessuna ricalibrazione", fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(percorso, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return percorso


def confronto(riepilogo, righe, saltate):
    d = os.path.join(CM.DIR_RISULTATI, "_confronto")
    os.makedirs(d, exist_ok=True)
    print()
    riga("=")
    print("  TABELLA COMPLESSIVA — da riportare in tesi")
    riga("=")
    stampa_tabella(righe)
    salva_tabella(righe, os.path.join(d, "tabella_tesi.csv"))
    plot_confronto(riepilogo, os.path.join(d, "confronto_configurazioni.png"))

    print()
    riga("-")
    print("  EFFETTI MARGINALI (differenze di medie, non test)")
    riga("-")

    def media(filtro, campo):
        v = [riepilogo[n][campo] for n in riepilogo if filtro(n)]
        v = [x for x in v if np.isfinite(x)]
        return float(np.mean(v)) if v else float("nan")

    for campo, et in (("sens_hsil", "sensibilita HSIL"),
                      ("spec_tessuto", "specificita su tessuto"),
                      ("miou", "mIoU 3 classi")):
        dv = (media(lambda n: n.startswith("vahadane_si"), campo) -
              media(lambda n: n.startswith("vahadane_no"), campo))
        dk = (media(lambda n: n.endswith("k3"), campo) -
              media(lambda n: n.endswith("k2"), campo))
        print(f"  {et:<26} Vahadane {dv:+.4f}    K3-K2 {dk:+.4f}")

    print()
    print("  Da leggere insieme agli intervalli della figura. Con questo numero")
    print("  di pazienti si sovrappongono quasi certamente: riportare la")
    print("  direzione e la sua incertezza, mai una conclusione di superiorita'.")
    print("  K=3.0 e' calibrato sulle statistiche di MTCHI: la sua cella non e'")
    print("  una validazione cieca e va etichettata come tale.")

    with open(os.path.join(d, "riepilogo.json"), "w", encoding="utf-8") as f:
        json.dump({"riepilogo": riepilogo, "righe": righe, "saltate": saltate,
                   "firma": CM.firma()}, f, indent=2, ensure_ascii=False,
                  default=float)
    print(f"\n  Confronto in: {d}")


# ====================================================================
# 3. EXPORT PER eval_task2.py
# ====================================================================
def esporta_dat(export, rep_estrazione, percorso):
    """
    Salva le mappe RoI gia' ricomposte, in spazio MTCHI (0-4), nel formato
    che eval_task2.py ufficiale si aspetta. Controllo indipendente: se la
    NLH-mIoU dello script MCPRL non coincide con la nostra, c'e' un errore.
    """
    import pickle
    info = {r["cartella"]: r for r in rep_estrazione["per_roi"]}
    with open(percorso, "wb") as f:
        pickle.dump({info[c]["nome"]: m for c, m in export.items()}, f)
    print(f"  Predizioni in spazio MTCHI: {percorso}")
    return percorso


# ====================================================================
# 4. ESEMPI QUALITATIVI
# ====================================================================
def candidato(voce, cm, truth3, pred3, vetro, n_annotati):
    """
    Record leggero: nessuna immagine resta in RAM. La selezione avviene su
    questi, e solo i tile scelti vengono riletti da disco.
    """
    if n_annotati == 0:
        return None
    iou = ME.iou_per_classe(cm)
    cop = [float((truth3 == c).sum()) / n_annotati
           for c in range(CM.NUM_CLASSI)]
    conf = {}
    for a in range(CM.NUM_CLASSI):
        na = int((truth3 == a).sum())
        if na == 0:
            continue
        for b in range(CM.NUM_CLASSI):
            if a != b:
                conf[f"{a}->{b}"] = float(
                    ((truth3 == a) & (pred3 == b)).sum()) / na
    return {"img": voce["img"], "msk": voce["msk"], "tile": voce["tile"],
            "cartella": voce["cartella"], "paziente": voce["paziente"],
            "iou": [None if np.isnan(v) else float(v) for v in iou],
            "copertura": cop, "confusioni": conf,
            "pixel_annotati": int(n_annotati)}


def _scegli_percentili(amm, chiave, max_per_roi):
    amm = sorted(amm, key=chiave)
    scelte, usati = [], {}
    for perc, etichetta in CM.PERCENTILI_ESEMPIO:
        i = int(round((perc / 100.0) * (len(amm) - 1)))
        for j in list(range(i, len(amm))) + list(range(i - 1, -1, -1)):
            d = amm[j]
            if usati.get(d["cartella"], 0) >= max_per_roi:
                continue
            if any(s["img"] == d["img"] for s in scelte):
                continue
            usati[d["cartella"]] = usati.get(d["cartella"], 0) + 1
            scelte.append({**d, "percentile": perc, "etichetta": etichetta,
                           "n_candidati": len(amm)})
            break
    return scelte


def seleziona(cand, cm_aggregata):
    cand = [c for c in cand if c is not None]
    fuori = []

    for c in range(CM.NUM_CLASSI):
        amm = [d for d in cand
               if d["copertura"][c] >= CM.FRAZIONE_MINIMA_ESEMPIO
               and d["iou"][c] is not None]
        if len(amm) < 3:
            continue
        for s in _scegli_percentili(amm, lambda d: d["iou"][c],
                                    CM.MAX_ESEMPI_PER_ROI):
            fuori.append({**s, "tipo": "classe", "classe": c,
                          "iou_classe": s["iou"][c]})

    # confusioni piu' frequenti nella matrice AGGREGATA: gli errori mostrati
    # sono quelli che pesano davvero, non quelli piu' vistosi
    cm = np.asarray(cm_aggregata, dtype=np.float64)
    coppie = []
    for a in range(CM.NUM_CLASSI):
        tot = cm[a].sum()
        if tot == 0:
            continue
        for b in range(CM.NUM_CLASSI):
            if a != b and cm[a, b] > 0:
                coppie.append((cm[a, b] / tot, a, b))
    coppie.sort(reverse=True)
    for _, a, b in coppie[:CM.N_ERRORI]:
        k = f"{a}->{b}"
        amm = [d for d in cand
               if d["confusioni"].get(k, 0.0) >= CM.FRAZIONE_MINIMA_ERRORE]
        if not amm:
            continue
        amm.sort(key=lambda d: -d["confusioni"][k])
        for d in amm:
            if any(s["img"] == d["img"] for s in fuori):
                continue
            fuori.append({**d, "tipo": "errore", "classe": a, "predetta": b,
                          "quota": d["confusioni"][k]})
            break
    return fuori


# --------------------------------------------------------------------
# RENDER — stessa disposizione, palette e stile di esempi.py
# --------------------------------------------------------------------
def _colora_maschera(idx, vetro=None):
    out = np.empty((*idx.shape, 3), dtype=np.uint8)
    out[:] = _NON_ANN
    valido = idx < CM.NUM_CLASSI
    out[valido] = _PAL[idx[valido]]
    if vetro is not None:
        out[valido & vetro] = _VETRO
    return out


def _mappa_errori(truth, pred):
    out = np.empty((*truth.shape, 3), dtype=np.uint8)
    out[:] = _NON_ANN
    valido = truth < CM.NUM_CLASSI
    corretto = valido & (pred == truth)
    out[corretto] = _OK
    out[valido & ~corretto] = _ERR
    return out


def _sovrapponi(rgb, colori, alpha=CM.ALPHA_OVERLAY):
    return (rgb.astype(np.float32) * (1 - alpha) +
            colori.astype(np.float32) * alpha).astype(np.uint8)


def _legenda(fig):
    voci = [Patch(facecolor=_PAL[i] / 255, edgecolor="black", linewidth=0.4,
                  label=CM.NOMI_CLASSI[i]) for i in range(CM.NUM_CLASSI)]
    voci.append(Patch(facecolor=_VETRO / 255, edgecolor="black",
                      linewidth=0.4, label="predetto vetro"))
    voci.append(Patch(facecolor=_NON_ANN / 255, edgecolor="black",
                      linewidth=0.4, label="non annotato"))
    fig.legend(handles=voci, loc="lower center", ncol=len(voci), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))


def _mascherata(pred, truth):
    """Fuori dall'annotazione non c'e' verita': si disegna grigio."""
    return np.where(truth < CM.NUM_CLASSI, pred, np.uint8(CM.IGNORE))


def _figura(rgb, truth, pred, vetro, titolo, percorso, sottotitolo=""):
    fig, ax = plt.subplots(1, 4, figsize=CM.FIGSIZE_ESEMPI)
    ax[0].imshow(rgb)
    ax[0].set_title("tile (H&E)", fontsize=10)
    ax[1].imshow(_sovrapponi(rgb, _colora_maschera(truth)))
    ax[1].set_title("ground truth MTCHI", fontsize=10)
    ax[2].imshow(_sovrapponi(rgb, _colora_maschera(_mascherata(pred, truth),
                                                   vetro)))
    ax[2].set_title("predizione", fontsize=10)
    ax[3].imshow(_sovrapponi(rgb, _mappa_errori(truth, pred)))
    ax[3].set_title("verde = corretto   rosa = errore   "
                    "grigio = non annotato", fontsize=9)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(titolo, fontsize=12, weight="bold")
    if sottotitolo:
        fig.text(0.5, 0.925, sottotitolo, ha="center", fontsize=9,
                 color="#444444")
    _legenda(fig)
    plt.tight_layout(rect=(0, 0.03, 1, 0.90))
    plt.savefig(percorso, dpi=CM.DPI_ESEMPI, bbox_inches="tight")
    plt.close(fig)


def _figura_decisioni(rgb, truth, pred_a, pred_c, titolo, percorso,
                      sottotitolo=""):
    fig, ax = plt.subplots(1, 4, figsize=CM.FIGSIZE_ESEMPI)
    ax[0].imshow(_sovrapponi(rgb, _colora_maschera(truth)))
    ax[0].set_title("ground truth MTCHI", fontsize=10)
    ax[1].imshow(_sovrapponi(rgb, _colora_maschera(_mascherata(pred_a, truth))))
    ax[1].set_title("decisione argmax", fontsize=10)
    ax[2].imshow(_sovrapponi(rgb, _colora_maschera(_mascherata(pred_c, truth))))
    ax[2].set_title("decisione a costo minimo", fontsize=10)
    diverso = pred_a != pred_c
    diff = np.empty((*truth.shape, 3), dtype=np.uint8)
    diff[:] = _NON_ANN
    diff[~diverso] = _OK
    diff[diverso] = _ERR
    ax[3].imshow(_sovrapponi(rgb, diff))
    ax[3].set_title(f"rosa = decisione diversa "
                    f"({100*diverso.mean():.1f}% dei pixel)", fontsize=9)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(titolo, fontsize=12, weight="bold")
    if sottotitolo:
        fig.text(0.5, 0.925, sottotitolo, ha="center", fontsize=9,
                 color="#444444")
    _legenda(fig)
    plt.tight_layout(rect=(0, 0.03, 1, 0.90))
    plt.savefig(percorso, dpi=CM.DPI_ESEMPI, bbox_inches="tight")
    plt.close(fig)


def _nome_file(s):
    tile = os.path.splitext(s["tile"])[0]
    if s["tipo"] == "errore":
        base = (f"errore_{CM.NOMI_CLASSI[s['classe']]}_verso_"
                f"{CM.NOMI_CLASSI[s['predetta']]}")
    elif s["tipo"] == "decisione":
        base = "decisione"
    else:
        base = (f"{CM.NOMI_CLASSI[s['classe']]}_p{s['percentile']:02d}_"
                f"{s['etichetta']}")
    return f"{base}__{s['cartella']}__{tile}.png"


def salva_esempi(cand, cm_aggregata, dir_uscita, dir_pred,
                 dir_pred_costo=None):
    """
    Immagini e predizioni si rileggono da disco solo per i tile scelti:
    nessuna inferenza viene ripetuta e niente resta in RAM.
    """
    scelte = seleziona(cand, cm_aggregata)

    if dir_pred_costo and CM.N_CONFRONTI_DECISIONE > 0:
        misurati = []
        for d in (x for x in cand if x is not None):
            pa = cv2.imread(os.path.join(dir_pred, d["cartella"], d["tile"]),
                            cv2.IMREAD_GRAYSCALE)
            pc = cv2.imread(os.path.join(dir_pred_costo, d["cartella"],
                                         d["tile"]), cv2.IMREAD_GRAYSCALE)
            if pa is None or pc is None:
                continue
            misurati.append((float((pa != pc).mean()), d))
        misurati.sort(key=lambda t: -t[0])
        for q, d in misurati[:CM.N_CONFRONTI_DECISIONE]:
            scelte.append({**d, "tipo": "decisione", "quota": q})

    if not scelte:
        print("   nessun esempio soddisfa la regola pre-specificata")
        return []

    os.makedirs(dir_uscita, exist_ok=True)
    manifest = []
    for s in scelte:
        rgb = cv2.cvtColor(cv2.imread(s["img"], cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
        t3 = ME.mappa_truth(cv2.imread(s["msk"], cv2.IMREAD_GRAYSCALE))
        p6 = cv2.imread(os.path.join(dir_pred, s["cartella"], s["tile"]),
                        cv2.IMREAD_GRAYSCALE)
        if p6 is None:
            manifest.append({**s, "file": None, "nota": "predizione assente"})
            continue
        p3, vetro = ME.mappa_predizione(p6, "negativo")
        nome = _nome_file(s)
        comune = (f"{s['cartella']} / {s['tile']}   paziente {s['paziente']}   "
                  f"annotato {100*s['pixel_annotati']/(CM.TILE**2):.0f}% del tile")

        if s["tipo"] == "decisione":
            p6c = cv2.imread(os.path.join(dir_pred_costo, s["cartella"],
                                          s["tile"]), cv2.IMREAD_GRAYSCALE)
            p3c, _ = ME.mappa_predizione(p6c, "negativo")
            _figura_decisioni(rgb, t3, p3, p3c, "Argmax contro costo minimo",
                              os.path.join(dir_uscita, nome), comune)
        elif s["tipo"] == "errore":
            tit = (f"Errore {CM.NOMI_CLASSI[s['classe']]} -> "
                   f"{CM.NOMI_CLASSI[s['predetta']]}")
            _figura(rgb, t3, p3, vetro, tit, os.path.join(dir_uscita, nome),
                    f"{comune}   quota confusa {100*s['quota']:.0f}%")
        else:
            tit = (f"{CM.NOMI_CLASSI[s['classe']]} — caso {s['etichetta']} "
                   f"(percentile {s['percentile']} su {s['n_candidati']} tile)")
            _figura(rgb, t3, p3, vetro, tit, os.path.join(dir_uscita, nome),
                    f"{comune}   IoU {s['iou_classe']:.3f}   "
                    f"copertura {100*s['copertura'][s['classe']]:.0f}%")
        manifest.append({**s, "file": nome})

    with open(os.path.join(dir_uscita, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump({"regola": {
            "frazione_minima": CM.FRAZIONE_MINIMA_ESEMPIO,
            "percentili": [list(p) for p in CM.PERCENTILI_ESEMPIO],
            "max_per_roi": CM.MAX_ESEMPI_PER_ROI,
            "n_errori": CM.N_ERRORI,
            "frazione_minima_errore": CM.FRAZIONE_MINIMA_ERRORE,
            "n_confronti_decisione": CM.N_CONFRONTI_DECISIONE},
            "scelte": manifest}, f, indent=2, ensure_ascii=False, default=float)
    print(f"   esempi: {sum(1 for m in manifest if m.get('file'))} figure "
          f"in {dir_uscita}")
    return manifest