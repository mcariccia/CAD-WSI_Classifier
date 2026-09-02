"""
esegui.py
====================================================================
Entry point. Orchestra: verifiche -> diagnosi -> fold -> training ->
riepilogo -> grafici -> report.

    python esegui.py

Per un giro rapido di verifica, in config.py:
    EPOCHS = 2
    FOLD_DA_ESEGUIRE = [0]
====================================================================
"""

import json
import math
import random

import numpy as np
import pandas as pd
import torch

import config as C
import dati as D
import metriche as MT
import diagnostica as DG
import grafici as GR
from addestramento import FoldTrainer


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def pulisci(o):
    """
    NaN e Inf non sono JSON valido. Python li rilegge, altri strumenti
    no: si convertono in null prima di salvare, cosi' i report restano
    leggibili anche fuori da Python.
    """
    if isinstance(o, dict):
        return {k: pulisci(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [pulisci(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return pulisci(o.tolist())
    return o


# ====================================================================
# RIEPILOGO
# ====================================================================
def riepilogo(risultati, diagnosi):
    print("\n" + "=" * 96)
    print("  RIEPILOGO CROSS-VALIDATION")
    print("=" * 96)
    print(f"  {'fold':>5} | {'ep':>3} | {'mIoULAST':>9} | {'accBil':>8} | "
          f"{'sensHSIL':>9} | {'specHSIL':>9} | {'recLSIL':>8} | "
          f"{'WSI':>4} | selezione")
    print("  " + "-" * 96)
    for r in risultati:
        rl = r["lsil_vs_hsil"]["recall_lsil"]
        s_rl = "n/a" if rl is None or np.isnan(rl) else f"{rl:.4f}"
        sel = "vincolo" if r["selezione"]["ammissibile"] else "RIPIEGO"
        print(f"  {r['fold']:>5} | {r['best_epoch']:>3} | "
              f"{r['miou_last_per_wsi']:>9.4f} | {r['accbil_last']:>8.4f} | "
              f"{r['hsil']['sensibilita']:>9.4f} | "
              f"{r['hsil']['specificita']:>9.4f} | {s_rl:>8} | "
              f"{len(r['wsi_val']):>4} | {sel}")

    n_ripiego = sum(1 for r in risultati if not r["selezione"]["ammissibile"])
    if n_ripiego:
        print(f"\n  ATTENZIONE: {n_ripiego} fold su {len(risultati)} non hanno")
        print("  prodotto NESSUNA epoca dentro l'operating point dichiarato.")
        print("  Per quei fold la selezione e' avvenuta sulla mIoU LAST e il")
        print("  confronto con gli altri non e' alla pari.")

    def ms(f):
        v = [f(r) for r in risultati]
        v = [x for x in v if x is not None and not np.isnan(x)]
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"),) * 2

    aggregati = {}
    print()
    for etichetta, chiave, f in [
            ("mIoU LAST per WSI", "miou_last_per_wsi",
             lambda r: r["miou_last_per_wsi"]),
            ("accuratezza bilanciata", "accbil_last", lambda r: r["accbil_last"]),
            ("sensibilita HSIL", "sens_hsil",
             lambda r: r["hsil"]["sensibilita"]),
            ("specificita HSIL", "spec_hsil",
             lambda r: r["hsil"]["specificita"]),
            ("Dice HSIL", "dice_hsil", lambda r: r["hsil"]["dice"]),
            ("recall LSIL (vs HSIL)", "recall_lsil",
             lambda r: r["lsil_vs_hsil"]["recall_lsil"]),
            ("mIoU classi fini", "miou_fine", lambda r: r["miou_fine"])]:
        m, s = ms(f)
        aggregati[chiave] = {"media": m, "std": s}
        print(f"  {etichetta:<26} {m:.4f} +- {s:.4f}")

    # ---- confusioni aggregate ----
    cm_fine = torch.tensor(np.sum([np.array(r["cm_fine"]) for r in risultati],
                                  axis=0))
    cm_tile = torch.tensor(np.sum([np.array(r["cm_tile"]) for r in risultati],
                                  axis=0))
    cm_last = MT.cm_to_last(cm_fine)

    print("\n  Confusione CLASSI FINI aggregata (% per riga):")
    MT.stampa_confusione(cm_fine, C.CLASS_NAMES)
    print()
    MT.stampa_per_classe(cm_fine, C.CLASS_NAMES)
    print("\n  Confusione LAST aggregata (% per riga):")
    MT.stampa_confusione(cm_last, C.NOMI_LAST)
    print("\n  Confusione per TILE aggregata (% per riga):")
    MT.stampa_confusione(cm_tile, C.CLASS_NAMES)

    print()
    MT.stampa_errori_principali(cm_fine, C.CLASS_NAMES, top=8,
                                titolo="Errori principali (classi fini)")
    print()
    MT.stampa_errori_principali(cm_last, C.NOMI_LAST, top=4,
                                titolo="Errori principali (livello LAST)")

    h = MT.metriche_hsil(cm_last)
    lh = MT.discriminazione_lsil_hsil(cm_last)
    accbil_agg = MT.accuratezza_bilanciata(cm_last, escludi=(C.IDX_GLASS_LAST,))
    print(f"\n  Accuratezza bilanciata LAST (GLASS escluso): {accbil_agg:.4f}")
    print("\n  DELIVERABLE PRIMARIO — HSIL vs non-HSIL (aggregato):")
    print(f"    sensibilita {h['sensibilita']:.4f}   "
          f"specificita {h['specificita']:.4f}   "
          f"precisione {h['precisione']:.4f}   Dice {h['dice']:.4f}")
    print(f"    pixel di tessuto predetti vetro: "
          f"{int(h['px_tessuto_predetti_vetro']):,}")
    print(f"\n  Confine LSIL/HSIL (esplorativo, {diagnosi['n_wsi_lsil_e_hsil']} "
          f"WSI di co-occorrenza):")
    print(f"    accuratezza {lh['accuratezza']:.4f}   "
          f"recall LSIL {lh['recall_lsil']:.4f}   "
          f"recall HSIL {lh['recall_hsil']:.4f}")

    tutte = {w: v for r in risultati for w, v in r["per_wsi"].items()}
    print(f"\n  mIoU LAST per WSI ({len(tutte)} WSI, ordinate):")
    for w, v in sorted(tutte.items(), key=lambda kv: kv[1]):
        print(f"    {w:<20}{v:.4f}")
    vals = list(tutte.values())
    print(f"    media {np.mean(vals):.4f}   mediana {np.median(vals):.4f}   "
          f"std {np.std(vals):.4f}")

    print("\n" + "=" * 96)
    print("  COME LEGGERE QUESTI NUMERI")
    print("=" * 96)
    for a in diagnosi.get("avvisi", []):
        print(f"  - {a}")
    print("  - Per le classi a bassa copertura la deviazione standard fra fold")
    print("    NON e' rumore attorno a un valore vero: ogni fold addestra un")
    print("    modello con un confine decisionale diverso. Riportare i valori")
    print("    per WSI singolarmente.")
    print("  - La VaLoss non e' confrontabile fra fold: e' una media per pixel")
    print("    su insiemi di validazione con composizione di classi diversa.")
    print("    Non usarla per nessuna decisione.")
    print("  - L'accuratezza grezza e' dominata dalle WSI piu' grandi: usare")
    print("    sempre quella bilanciata.")
    print("  - Il confine LSIL/HSIL va riportato come esplorativo finche' la")
    print(f"    co-occorrenza resta sotto le {C.SOGLIA_COOCCORRENZA} WSI.")
    print("=" * 96)

    return aggregati, cm_fine, cm_last, cm_tile, tutte, accbil_agg


# ====================================================================
# MAIN
# ====================================================================
def main():
    set_seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Esperimento: {C.firma_esperimento()}")
    if device.type != "cuda":
        print("ATTENZIONE: nessuna GPU rilevata, il training sara' molto lento.")

    print("\n--- VERIFICHE ---")
    DG.verifica_cartelle()
    DG.verifica_moduli()
    DG.verifica_riempimento()
    DG.verifica_maschere()

    print("\n--- DATI ---")
    campioni_totali = D.elenca_campioni()
    if not campioni_totali:
        print("Nessuna tile trovata. Verificare CARTELLA_IMG in config.py.")
        return

    print("\n--- FILTRO DATASET (pool per la cross-validation) ---")
    try:
        df_cv = pd.read_csv(C.PATH_TRAIN_CSV, index_col=False, encoding="utf-8")
        wsi_ammessi = set(df_cv["WSI"].tolist())
        campioni = [c for c in campioni_totali if c[2] in wsi_ammessi]
        n_disco = len({c[2] for c in campioni_totali})
        print(f"   WSI totali su disco : {n_disco}")
        print(f"   WSI ammesse alla CV : {len(wsi_ammessi)}")
        print(f"   WSI di test escluse : {n_disco - len(wsi_ammessi)}")
        print(f"   Tile ridotte da {len(campioni_totali)} a {len(campioni)}")
    except Exception as e:
        print(f"ERRORE nella lettura di {C.PATH_TRAIN_CSV}: {e}")
        return
    if not campioni:
        print("Nessuna tile dopo il filtro: controllare i nomi WSI nel CSV.")
        return

    diagnosi_vetro = DG.verifica_soglia_vetro(campioni)

    comp = D.composizione(campioni)
    px_wsi, presenza, non_validabili, diagnosi = DG.analizza_dataset(campioni,
                                                                     comp)
    diagnosi["pseudo_etichetta_vetro"] = diagnosi_vetro

    n_wsi = len({c[2] for c in campioni})
    if C.N_FOLD > n_wsi:
        raise RuntimeError(
            f"N_FOLD={C.N_FOLD} ma ci sono solo {n_wsi} WSI: impossibile "
            f"creare fold disgiunti. Ridurre N_FOLD in config.py.")

    folds, assegnazione = D.crea_fold(campioni, presenza)
    D.stampa_fold(folds, assegnazione, presenza)

    da_eseguire = (C.FOLD_DA_ESEGUIRE if C.FOLD_DA_ESEGUIRE is not None
                   else range(len(folds)))

    risultati, architettura = [], None
    for k in da_eseguire:
        idx_tr, idx_va = folds[k]
        set_seed(C.SEED + k)
        trainer = FoldTrainer(k, campioni, comp, idx_tr, idx_va, device,
                              classi_escluse=non_validabili)
        architettura = type(trainer.model).__name__
        r = trainer.train()
        if r:
            risultati.append(r)
        # Salvataggio incrementale: se il run si interrompe, i fold gia'
        # completati non si perdono.
        with open(C.path_metrica(C.NOME_STORIA), "w", encoding="utf-8") as f:
            json.dump(pulisci(risultati), f, indent=2, ensure_ascii=False)

    if not risultati:
        print("\nNessun fold completato.")
        return

    (aggregati, cm_fine, cm_last, cm_tile,
     per_wsi, accbil_agg) = riepilogo(risultati, diagnosi)

    prodotti = GR.salva_tutto(risultati, cm_fine, cm_last, cm_tile, per_wsi)
    print("\n  Grafici e tabelle prodotti:")
    for nome, percorso in prodotti.items():
        print(f"    {nome:<22}{percorso}")

    report = {
        "config": {
            "nome_run": C.NOME_RUN,
            "architettura": architettura,
            "encoder": C.ENCODER,
            "modalita_classi": C.MODALITA_CLASSI,
            "normalizzazione": C.LIVELLO_NORMALIZZAZIONE,
            "cartella_img": C.CARTELLA_IMG,
            "tile": C.TILE,
            "batch_size": C.BATCH_SIZE,
            "n_fold": C.N_FOLD,
            "epochs": C.EPOCHS,
            "patience": C.PATIENCE,
            "seed": C.SEED,
            "cutmix": C.USA_CUTMIX,
            "clip_pesi": list(C.CLIP_PESI),
            "sampling_alpha": C.SAMPLING_ALPHA,
            "wsi_balance": C.WSI_BALANCE,
            "hed": C.parametri_hed(),
            "vetro": C.parametri_vetro(),
            "loss": C.parametri_loss(),
            "selezione": C.parametri_selezione(),
            "firma": C.firma_esperimento(),
        },
        "diagnosi": diagnosi,
        "aggregati": aggregati,
        "cm_fine_aggregata": cm_fine.tolist(),
        "cm_last_aggregata": cm_last.tolist(),
        "cm_tile_aggregata": cm_tile.tolist(),
        "hsil_aggregato": MT.metriche_hsil(cm_last),
        "accbil_last_aggregata": accbil_agg,
        "errori_principali_fine": MT.errori_principali(cm_fine, C.CLASS_NAMES,
                                                       top=12),
        "errori_principali_last": MT.errori_principali(cm_last, C.NOMI_LAST,
                                                       top=6),
        "lsil_vs_hsil_aggregato": MT.discriminazione_lsil_hsil(cm_last),
        "per_wsi": per_wsi,
        "cv": risultati,
    }
    with open(C.path_metrica(C.NOME_REPORT), "w", encoding="utf-8") as f:
        json.dump(pulisci(report), f, indent=2, ensure_ascii=False)

    print(f"\nReport completo : {C.path_metrica(C.NOME_REPORT)}")
    print(f"Storia per fold : {C.path_metrica(C.NOME_STORIA)}")
    print(f"Modelli         : {C.DIR_MODELLI}")
    print(f"Grafici         : {C.DIR_GRAFICI}")


if __name__ == "__main__":
    main()