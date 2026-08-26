"""
grafici.py
====================================================================
Curve di apprendimento e matrici di confusione come immagini.

La matrice di confusione stampata a terminale dice DOVE sbaglia il
modello, ma su 6x6 numeri e' faticosa da leggere. La versione grafica
rende immediato individuare i blocchi di errore: se Stroma e Mucosa si
confondono, si vede una macchia fuori diagonale in quel punto.

Tutte le matrici sono normalizzate per RIGA, cioe' mostrano la
percentuale dei pixel di ciascuna classe VERA. Una riga somma sempre a
100%. Si legge: "dei pixel che erano Stroma, il 60% e' stato predetto
Mucosa".
====================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C


def _disegna_matrice(ax, cm, nomi, titolo, cmap="Blues"):
    """
    Heatmap annotata, normalizzata per riga.

    Il colore segue la percentuale, il numero nella cella e' quella
    percentuale. Sotto ogni nome di classe, fra parentesi, il numero
    assoluto di pixel veri: serve a capire se una riga con percentuali
    estreme poggia su tanti o pochissimi dati.
    """
    cm = np.asarray(cm, dtype=np.float64)
    righe = cm.sum(1, keepdims=True)
    norm = 100.0 * cm / np.maximum(righe, 1)

    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=100)
    n = len(nomi)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(nomi, rotation=45, ha="right", fontsize=8)
    etichette_y = [f"{nomi[i]}\n({int(righe[i, 0]):,})" for i in range(n)]
    ax.set_yticklabels(etichette_y, fontsize=7)
    ax.set_xlabel("predetta", fontsize=9)
    ax.set_ylabel("vera (n pixel)", fontsize=9)
    ax.set_title(titolo, fontsize=10, weight="bold")

    for i in range(n):
        for j in range(n):
            v = norm[i, j]
            if righe[i, 0] == 0:
                testo, colore = "-", "#999999"
            else:
                testo = f"{v:.1f}"
                colore = "white" if v > 55 else "black"
            ax.text(j, i, testo, ha="center", va="center",
                    fontsize=8, color=colore)
    return im


def plot_confusioni(cm_fine, cm_last, cm_tile, nome_file="confusioni.png",
                    sottotitolo=""):
    """
    Tre pannelli affiancati:

      1. CLASSI FINI, per pixel — diagnostico. Mostra le confusioni
         morfologiche reali, comprese quelle clinicamente irrilevanti.
      2. LAST, per pixel — clinico. Le confusioni dentro il gruppo
         negativo sono sparite: quello che resta fuori diagonale sono
         errori che cambiano la gestione della paziente.
      3. CLASSI FINI, per TILE — coerente con annotazioni region-level.
         Se questa e' molto migliore della prima, il modello classifica
         bene e localizza male.
    """
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.8))
    _disegna_matrice(ax[0], cm_fine, C.CLASS_NAMES,
                     "Classi fini — per pixel (% per riga)")
    _disegna_matrice(ax[1], cm_last, C.NOMI_LAST,
                     "Livello LAST — per pixel (% per riga)", cmap="Greens")
    _disegna_matrice(ax[2], cm_tile, C.CLASS_NAMES,
                     "Classi fini — per TILE (% per riga)", cmap="Purples")

    titolo = "Matrici di confusione"
    if sottotitolo:
        titolo += f" — {sottotitolo}"
    fig.suptitle(titolo, fontsize=13, weight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    percorso = C.path_grafico(nome_file)
    plt.savefig(percorso, dpi=150)
    plt.close(fig)
    return percorso


def plot_confusioni_per_fold(risultati, nome_file="confusioni_per_fold.png"):
    """
    Una matrice LAST per fold, affiancate.

    Serve a vedere se il modello sbaglia sempre allo stesso modo o se
    l'errore dipende da quali WSI sono capitate in validation. Nel
    secondo caso la media fra fold non descrive nessun modello reale.
    """
    n = len(risultati)
    if n == 0:
        return None
    fig, ax = plt.subplots(1, n, figsize=(4.6 * n, 4.8), squeeze=False)
    for i, r in enumerate(risultati):
        _disegna_matrice(ax[0][i], np.array(r["cm_last"]), C.NOMI_LAST,
                         f"fold {r['fold']}  ({len(r['wsi_val'])} WSI)",
                         cmap="Greens")
    fig.suptitle("Confusione LAST per fold (% per riga)",
                 fontsize=13, weight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    percorso = C.path_grafico(nome_file)
    plt.savefig(percorso, dpi=150)
    plt.close(fig)
    return percorso


def plot_per_wsi(per_wsi, nome_file="miou_per_wsi.png"):
    """
    Barre orizzontali della mIoU LAST di ciascuna WSI, ordinate.

    E' il grafico piu' onesto del lotto: con classi presenti in poche
    WSI, la media non descrive nulla e questa distribuzione e' il vero
    risultato. Le barre rosse (sotto 0,30) sono le WSI su cui il
    modello fallisce e vanno discusse singolarmente.
    """
    if not per_wsi:
        return None
    voci = sorted(per_wsi.items(), key=lambda kv: kv[1])
    nomi = [k for k, _ in voci]
    vals = [v for _, v in voci]
    colori = ["#C44E52" if v < 0.30 else
              "#DD8452" if v < 0.50 else "#4C72B0" for v in vals]

    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.26 * len(nomi))))
    ax.barh(range(len(nomi)), vals, color=colori)
    ax.set_yticks(range(len(nomi)))
    ax.set_yticklabels(nomi, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("mIoU LAST (classi presenti)")
    media = float(np.mean(vals))
    ax.axvline(media, color="black", ls="--", lw=1.2,
               label=f"media {media:.3f}")
    ax.axvline(float(np.median(vals)), color="gray", ls=":", lw=1.2,
               label=f"mediana {np.median(vals):.3f}")
    ax.legend(fontsize=8)
    ax.set_title("mIoU per WSI — rosso sotto 0,30", fontsize=11, weight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    percorso = C.path_grafico(nome_file)
    plt.savefig(percorso, dpi=150)
    plt.close(fig)
    return percorso


def plot_curve(risultati, nome_file=None):
    """Loss, mIoU e metriche cliniche, una linea per fold."""
    nome_file = nome_file or C.NOME_PLOT
    storie = [r["history"] for r in risultati if r]
    if not storie:
        return None

    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    for s in storie:
        ep = range(1, len(s["train_loss"]) + 1)
        ax[0].plot(ep, s["train_loss"], "b-", lw=1, alpha=0.45)
        ax[0].plot(ep, s["val_loss"], "r--", lw=1, alpha=0.45)
        ax[1].plot(ep, s["train_miou"], "b-", lw=1, alpha=0.35)
        ax[1].plot(ep, s["val_miou_last"], "g-", lw=1.6, alpha=0.75)
        ax[2].plot(ep, s["val_sens_hsil"], "m-", lw=1.6, alpha=0.75)
        ax[2].plot(ep, s["val_spec_hsil"], "c--", lw=1.4, alpha=0.6)

    ax[0].set_title("Loss (blu = train, rosso = val)", weight="bold")
    ax[0].set_ylabel("Loss")
    ax[1].set_title("mIoU (blu = train classi fini, verde = val LAST per WSI)",
                    weight="bold")
    ax[1].set_ylabel("mIoU")
    ax[1].set_ylim(0, 1)
    ax[2].set_title("HSIL vs non-HSIL "
                    "(magenta = sensibilita', ciano = specificita')",
                    weight="bold")
    ax[2].set_ylim(0, 1)
    for a in ax:
        a.set_xlabel("Epoche")
        a.grid(alpha=0.3)

    plt.tight_layout()
    percorso = C.path_grafico(nome_file)
    plt.savefig(percorso, dpi=150)
    plt.close(fig)
    return percorso


def salva_csv(cm, nomi, nome_file):
    """
    Confusione in CSV con intestazioni, per aprirla in Excel o
    importarla in tesi senza riscriverla a mano.
    """
    cm = np.asarray(cm)
    percorso = C.path_metrica(nome_file)
    with open(percorso, "w", encoding="utf-8") as f:
        f.write("vera\\predetta," + ",".join(nomi) + ",totale\n")
        for i, nome in enumerate(nomi):
            f.write(nome + "," + ",".join(str(int(x)) for x in cm[i]) +
                    f",{int(cm[i].sum())}\n")
        f.write("totale," +
                ",".join(str(int(x)) for x in cm.sum(0)) +
                f",{int(cm.sum())}\n")
    return percorso


def salva_tutto(risultati, cm_fine, cm_last, cm_tile, per_wsi):
    """Produce e salva tutti i grafici e i CSV; ritorna i percorsi."""
    prodotti = {
        "curve": plot_curve(risultati),
        "confusioni": plot_confusioni(
            cm_fine, cm_last, cm_tile,
            sottotitolo=f"aggregato su {len(risultati)} fold"),
        "confusioni_per_fold": plot_confusioni_per_fold(risultati),
        "miou_per_wsi": plot_per_wsi(per_wsi),
        "csv_fine": salva_csv(cm_fine, C.CLASS_NAMES, "confusione_fine.csv"),
        "csv_last": salva_csv(cm_last, C.NOMI_LAST, "confusione_last.csv"),
        "csv_tile": salva_csv(cm_tile, C.CLASS_NAMES, "confusione_tile.csv"),
    }
    return {k: v for k, v in prodotti.items() if v}