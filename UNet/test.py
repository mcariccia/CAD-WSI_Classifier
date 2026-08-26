"""
valuta_test.py
====================================================================
Valutazione FINALE sulle WSI tenute fuori dalla cross-validation.

DA ESEGUIRE UNA VOLTA SOLA, alla fine.
---------------------------------------
Se si usa il test set per scegliere iperparametri, o si rilancia
cambiando qualcosa dopo aver visto i numeri, smette di essere un test
e diventa un secondo validation set. La stima che produce non e' piu'
una stima di generalizzazione.

Se serve provare varianti, si provano in cross-validation.

PERCHE' NON SI SCEGLIE "IL FOLD MIGLIORE"
------------------------------------------
I punteggi dei fold sono punteggi di VALIDATION. Prenderne il massimo
e riportarlo e' selezione sui dati di validazione. Sul run a 34 WSI:

    fold 0   0.7573   <-- il massimo
    fold 1   0.7559
    fold 3   0.7161
    fold 2   0.7088
    fold 4   0.6705
    media    0.7217   <-- la stima onesta

Riportare 0.7573 gonfierebbe di 3.6 punti. E i cinque numeri non sono
nemmeno confrontabili fra loro: il fold 0 aveva in validation Prova 2
(87% stroma, facile), il fold 4 aveva Prova 18 e Prova 46 (le due CIN1
piu' difficili). Il fold 0 non e' il modello migliore, e' il fold piu'
fortunato.

DUE MODALITA'
--------------
  ensemble  : media delle softmax dei modelli dei fold. Nessuno dei
              modelli ha mai visto le WSI di test, quindi e' lecito.
              Guadagno tipico 1-3 punti, a costo zero.
  singoli   : ogni modello valutato separatamente. Serve a mostrare
              che l'ensemble non e' un artificio e a quantificare la
              variabilita' fra modelli sullo STESSO test set — che e'
              informazione diversa dalla variabilita' fra fold.

Vengono riportate entrambe.

INTERVALLI DI CONFIDENZA
-------------------------
Con poche WSI di test qualunque numero puntuale e' fragile. Si calcola
un intervallo bootstrap ricampionando le WSI (non i pixel: i pixel
dentro una WSI sono fortemente correlati e ricampionarli produrrebbe
intervalli assurdamente stretti).

Uso
----
    python valuta_test.py
====================================================================
"""

import os
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config as C
import dati as D
import modello as M
import metriche as MT

# ====================================================================
# PARAMETRI
# ====================================================================
CSV_TEST = r"E:\Tirocinio\Progetto\UNet\Dataset_UNet\test_split.csv"
NOME_REPORT = "report_test.json"

USA_TTA = True          
N_BOOTSTRAP = 2000
SEED_BOOTSTRAP = 42
# ====================================================================


def carica_modelli(device):
    """Carica i checkpoint di tutti i fold disponibili."""
    modelli, nomi = [], []
    for k in range(C.N_FOLD):
        path = C.path_modello(k)
        if not os.path.exists(path):
            print(f"   fold {k}: checkpoint mancante, salto ({path})")
            continue
        m = M.build_model(device, verbose=False)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        modelli.append(m)
        nomi.append(f"fold {k}")
    if not modelli:
        raise RuntimeError(f"Nessun checkpoint trovato in {C.DIR_MODELLI}")
    print(f"   modelli caricati: {len(modelli)}  ({', '.join(nomi)})")
    return modelli, nomi


@torch.no_grad()
def _softmax_batch(modelli, images, amp, tta=USA_TTA):
    """
    Media delle softmax sui modelli e sulle varianti TTA.

    Si mediano le PROBABILITA' e non i logit: i logit di modelli
    diversi non sono su scale confrontabili, mentre le probabilita'
    sono tutte normalizzate a 1 e la media resta una distribuzione
    valida.
    """
    somma = None
    n = 0
    varianti = [images]
    if tta:
        varianti += [torch.flip(images, [3]), torch.flip(images, [2]),
                     torch.flip(images, [2, 3])]
    for m in modelli:
        for i, v in enumerate(varianti):
            with torch.amp.autocast("cuda", enabled=amp):
                p = m(v).float().softmax(1)
            if i == 1:
                p = torch.flip(p, [3])
            elif i == 2:
                p = torch.flip(p, [2])
            elif i == 3:
                p = torch.flip(p, [2, 3])
            somma = p if somma is None else somma + p
            n += 1
    return somma / n


@torch.no_grad()
def valuta(modelli, loader, device, amp, etichetta=""):
    """-> (cm_fine, cm_tile, {wsi: cm_fine})"""
    cm = MT.nuova_cm(device=device)
    cm_tile = MT.nuova_cm(device=device)
    cm_wsi = {}
    desc = f"  {etichetta}" if etichetta else "  test"
    for images, masks, wsis in tqdm(loader, desc=desc, ncols=78):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        pred = _softmax_batch(modelli, images, amp).argmax(1)
        MT.aggiorna_cm(cm, pred, masks)
        MT.aggiorna_cm_tile(cm_tile, pred, masks)
        for w in dict.fromkeys(wsis):
            sel = [j for j, x in enumerate(wsis) if x == w]
            cm_wsi.setdefault(w, MT.nuova_cm(device=device))
            MT.aggiorna_cm(cm_wsi[w], pred[sel], masks[sel])
    return (cm.cpu(), cm_tile.cpu(), {w: c.cpu() for w, c in cm_wsi.items()})


def bootstrap_wsi(cm_wsi, funzione, n=N_BOOTSTRAP, seed=SEED_BOOTSTRAP):
    """
    Intervallo di confidenza al 95% ricampionando le WSI con
    reinserimento.

    Si ricampionano le WSI e NON i pixel. I pixel dentro una WSI sono
    fortemente correlati (stesso paziente, stesso taglio, stessa
    colorazione): trattarli come indipendenti produrrebbe intervalli
    strettissimi e falsi. L'unita' statistica e' il vetrino.
    """
    nomi = sorted(cm_wsi.keys())
    if len(nomi) < 3:
        return None
    rng = np.random.default_rng(seed)
    valori = []
    for _ in range(n):
        scelte = rng.choice(len(nomi), len(nomi), replace=True)
        somma = sum(cm_wsi[nomi[i]] for i in scelte)
        v = funzione(somma)
        if not np.isnan(v):
            valori.append(v)
    if not valori:
        return None
    return (float(np.percentile(valori, 2.5)),
            float(np.percentile(valori, 97.5)))


def riassumi(cm, cm_tile, cm_wsi, titolo):
    """Stampa e restituisce il blocco di metriche per una configurazione."""
    cm_last = MT.cm_to_last(cm)
    hsil = MT.metriche_hsil(cm_last)
    lh = MT.discriminazione_lsil_hsil(cm_last)

    print("\n" + "=" * 78)
    print(f"  {titolo}")
    print("=" * 78)
    MT.stampa_confusione(cm, C.CLASS_NAMES, "confusione classi fini (% per riga)")
    print()
    MT.stampa_per_classe(cm, C.CLASS_NAMES)
    print()
    MT.stampa_confusione(cm_last, C.NOMI_LAST, "confusione LAST (% per riga)")
    print()
    MT.stampa_errori_principali(cm, C.CLASS_NAMES, top=6)

    ic_sens = bootstrap_wsi(cm_wsi,
                            lambda c: MT.metriche_hsil(MT.cm_to_last(c))["sensibilita"])
    ic_spec = bootstrap_wsi(cm_wsi,
                            lambda c: MT.metriche_hsil(MT.cm_to_last(c))["specificita"])
    ic_miou = bootstrap_wsi(cm_wsi,
                            lambda c: MT.macro_presenti(MT.cm_to_last(c),
                                                        escludi=(C.IDX_GLASS_LAST,)))

    def _ic(t):
        return f"  IC95% [{t[0]:.3f}, {t[1]:.3f}]" if t else ""

    print(f"\n  DELIVERABLE PRIMARIO — HSIL vs non-HSIL")
    print(f"    sensibilita  {hsil['sensibilita']:.4f}{_ic(ic_sens)}")
    print(f"    specificita  {hsil['specificita']:.4f}{_ic(ic_spec)}")
    print(f"    precisione   {hsil['precisione']:.4f}")
    print(f"    Dice         {hsil['dice']:.4f}")
    print(f"    px tessuto predetti vetro: "
          f"{int(hsil['px_tessuto_predetti_vetro']):,}")

    accbil = MT.accuratezza_bilanciata(cm_last, escludi=(C.IDX_GLASS_LAST,))
    print(f"\n  accuratezza bilanciata LAST (GLASS escluso): {accbil:.4f}")
    print(f"  accuratezza per tile: {MT.accuratezza(cm_tile):.4f}"
          f"   bilanciata: {MT.accuratezza_bilanciata(cm_tile):.4f}")
    print(f"\n  Confine LSIL/HSIL (esplorativo): accuratezza {lh['accuratezza']:.4f}"
          f"   recall LSIL {lh['recall_lsil']:.4f}   recall HSIL {lh['recall_hsil']:.4f}")

    per_wsi = {}
    print(f"\n  Per WSI (mIoU LAST sulle classi presenti){_ic(ic_miou)}:")
    for w, c in sorted(cm_wsi.items(),
                       key=lambda kv: MT.macro_presenti(
                           MT.cm_to_last(kv[1]), escludi=(C.IDX_GLASS_LAST,))):
        cl = MT.cm_to_last(c)
        v = MT.macro_presenti(cl, escludi=(C.IDX_GLASS_LAST,))
        per_wsi[w] = float(v)
        rec = MT.recall_per_classe(cl)
        n = cl.double().sum(1)
        dett = "  ".join(
            f"{C.NOMI_LAST[i]} {100*rec[i]:.0f}%" if n[i] > 0 else f"{C.NOMI_LAST[i]} -"
            for i in range(1, 4))
        print(f"    {w:<14}{v:.4f}   {dett}")
    vals = list(per_wsi.values())
    print(f"    media {np.mean(vals):.4f}   mediana {np.median(vals):.4f}   "
          f"std {np.std(vals):.4f}   (n = {len(vals)} WSI)")

    return {
        "hsil": hsil,
        "lsil_vs_hsil": lh,
        "accbil_last": float(accbil),
        "accbil_tile": float(MT.accuratezza_bilanciata(cm_tile)),
        "miou_last_per_wsi": float(np.mean(vals)),
        "ic95_sensibilita": ic_sens,
        "ic95_specificita": ic_spec,
        "ic95_miou": ic_miou,
        "iou_fine": [None if np.isnan(v) else float(v)
                     for v in MT.iou_per_classe(cm).tolist()],
        "recall_fine": MT.recall_per_classe(cm),
        "cm_fine": cm.tolist(),
        "cm_last": cm_last.tolist(),
        "cm_tile": cm_tile.tolist(),
        "per_wsi": per_wsi,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print("\n" + "!" * 78)
    print("  Questa valutazione va eseguita UNA VOLTA SOLA.")
    print("  Rilanciarla dopo aver cambiato qualcosa in base ai suoi risultati")
    print("  trasforma il test set in un secondo validation set.")
    print("!" * 78)

    print("\n--- DATI DI TEST ---")
    campioni_tot = D.elenca_campioni()
    df = pd.read_csv(CSV_TEST)
    wsi_test = set(df["WSI"].tolist())
    campioni = [c for c in campioni_tot if c[2] in wsi_test]
    if not campioni:
        print(f"Nessuna tile di test trovata. Verificare CSV_TEST: {CSV_TEST}")
        return

    wsi_trovate = sorted({c[2] for c in campioni})
    print(f"   WSI di test attese : {len(wsi_test)}")
    print(f"   WSI di test trovate: {len(wsi_trovate)}  ({', '.join(wsi_trovate)})")
    print(f"   Tile di test       : {len(campioni):,}")
    mancanti = wsi_test - set(wsi_trovate)
    if mancanti:
        print(f"   ATTENZIONE: WSI nel CSV ma non su disco: {sorted(mancanti)}")

    path_diag = C.path_metrica(C.NOME_DIAGNOSI)
    if os.path.exists(path_diag):
        try:
            wsi_cv = set(json.load(open(path_diag, encoding="utf-8"))["px_per_wsi"])
            overlap = wsi_cv & wsi_test
            if overlap:
                raise RuntimeError(
                    f"LEAKAGE: {sorted(overlap)} compaiono sia nella "
                    f"cross-validation sia nel test set.")
            print(f"   controllo leakage: nessuna sovrapposizione con le "
                  f"{len(wsi_cv)} WSI della CV")
        except (KeyError, ValueError):
            print("   (impossibile verificare il leakage dalla diagnosi)")

    ds = D.DatasetPrivato(campioni, D.transforms_val())
    loader = torch.utils.data.DataLoader(
        ds, batch_size=C.BATCH_SIZE, shuffle=False,
        num_workers=C.NUM_WORKERS, pin_memory=True)

    print("\n--- MODELLI ---")
    modelli, nomi = carica_modelli(device)
    amp = (device.type == "cuda")

    # ---------------- ensemble ----------------
    cm, cm_tile, cm_wsi = valuta(modelli, loader, device, amp,
                                 f"ensemble ({len(modelli)} modelli)")
    ris_ens = riassumi(cm, cm_tile, cm_wsi,
                       f"TEST — ENSEMBLE DI {len(modelli)} MODELLI"
                       f"{' + TTA' if USA_TTA else ''}")

    # ---------------- modelli singoli ----------------
    print("\n" + "=" * 78)
    print("  TEST — MODELLI SINGOLI (stesso test set)")
    print("=" * 78)
    print(f"  {'modello':<12}{'sensHSIL':>10}{'specHSIL':>10}"
          f"{'accBil':>9}{'mIoULAST':>10}")
    print("  " + "-" * 52)
    singoli = {}
    for m, nome in zip(modelli, nomi):
        c, ct, cw = valuta([m], loader, device, amp, nome)
        cl = MT.cm_to_last(c)
        h = MT.metriche_hsil(cl)
        vals = [MT.macro_presenti(MT.cm_to_last(x), escludi=(C.IDX_GLASS_LAST,))
                for x in cw.values()]
        vals = [v for v in vals if not np.isnan(v)]
        ab = MT.accuratezza_bilanciata(cl, escludi=(C.IDX_GLASS_LAST,))
        singoli[nome] = {"sensibilita": h["sensibilita"],
                         "specificita": h["specificita"],
                         "accbil_last": float(ab),
                         "miou_last_per_wsi": float(np.mean(vals)) if vals else None}
        print(f"  {nome:<12}{h['sensibilita']:>10.4f}{h['specificita']:>10.4f}"
              f"{ab:>9.4f}{np.mean(vals):>10.4f}")

    ms = np.mean([v["sensibilita"] for v in singoli.values()])
    guad = ris_ens["hsil"]["sensibilita"] - ms
    print(f"\n  media dei singoli: {ms:.4f}   ensemble: "
          f"{ris_ens['hsil']['sensibilita']:.4f}   ({guad:+.4f})")
    print("  La variabilita' qui e' fra MODELLI sullo stesso test set: e'")
    print("  informazione diversa dalla variabilita' fra fold, che mescolava")
    print("  modelli diversi E insiemi di validation diversi.")

    report = {
        "csv_test": CSV_TEST,
        "wsi_test": wsi_trovate,
        "n_tile": len(campioni),
        "n_modelli": len(modelli),
        "tta": USA_TTA,
        "n_bootstrap": N_BOOTSTRAP,
        "ensemble": ris_ens,
        "singoli": singoli,
    }
    with open(C.path_metrica(NOME_REPORT), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport di test: {C.path_metrica(NOME_REPORT)}")

    print("\n" + "=" * 78)
    print("  COME RIPORTARLO")
    print("=" * 78)
    print("  - Il numero da mettere in tesi e' quello dell'ENSEMBLE con il suo")
    print("    intervallo bootstrap, non il migliore dei singoli.")
    print(f"  - Con {len(wsi_trovate)} WSI di test l'intervallo e' largo: va scritto,")
    print("    non nascosto. Un IC ampio dichiarato e' rigore; un numero")
    print("    puntuale su 6 vetrini presentato come definitivo non lo e'.")
    print("  - La cross-validation resta la stima di generalizzazione; il test")
    print("    e' la conferma su dati mai toccati. Vanno riportati entrambi, e")
    print("    se concordano e' un risultato di per se'.")
    print("=" * 78)


if __name__ == "__main__":
    main()