"""
test_esterno.py
====================================================================
Validazione esterna del modello privato sui tile di MTCHI Task 2.

Disegno fattoriale 2x2:

                    K = 2.0            K = 3.0
    Vahadane no   vahadane_no_k2     vahadane_no_k3
    Vahadane si   vahadane_si_k2     vahadane_si_k3

MODELLI CONGELATI. Nessun peso, nessuna soglia, nessuna selezione di
checkpoint dipende da queste immagini. Da eseguire UNA volta per
configurazione: se la si usa per scegliere qualcosa smette di essere una
validazione esterna.

PERCHE' PER TILE E NON SU RoI INTERE
-------------------------------------
Il test interno (test.py) valuta per tile 512. Tassellando anche MTCHI i
due test usano lo stesso protocollo e i numeri sono confrontabili. Una
finestra scorrevole darebbe piu' contesto ai bordi, ma introdurrebbe una
differenza di metodo fra test interno ed esterno, che vale piu' del
guadagno.

I tile di bordo dei frammenti epiteliali hanno contesto troncato e il
modello ci sbaglia di piu'. Succede identico sul test interno, quindi
l'errore e' comune ai due e il confronto resta pulito.

COSA SI RIPORTA
----------------
Sensibilita' e specificita' NON dipendono dalla prevalenza: sono le uniche
metriche confrontabili fra test interno e MTCHI, che hanno composizione
completamente diversa. Precisione, Dice e IoU stanno nel report ma vanno
letti solo accanto alla composizione.

Asimmetria da dichiarare in tesi: MTCHI annota SOLO l'epitelio, quindi la
specificita' qui e' calcolata contro epitelio Normal. Quella interna
include stroma (44,7% dei pixel) e ghiandole, che sono negativi facili.
Stesso nome, due quantita' diverse, e quella di MTCHI e' piu' difficile.

USO
    python test_mtchi.py                     tutte e quattro
    python test_mtchi.py vahadane_si_k2      una sola
====================================================================
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
import shutil

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config_esterno as CM
import metriche_esterno as ME
import grafici_esterno as FG

# --------------------------------------------------------------------
# Pipeline privata: il modello va costruito dallo STESSO codice che lo ha
# addestrato, non da una copia.
# --------------------------------------------------------------------
if CM.DIR_PIPELINE not in sys.path:
    sys.path.insert(0, CM.DIR_PIPELINE)
import config as CP        # noqa: E402
import modello as MP       # noqa: E402


def riga(c="-", n=78):
    print(c * n)


def titolo(t):
    print()
    riga("=")
    print("  " + t)
    riga("=")


# ====================================================================
# 1. VERIFICHE PRELIMINARI
# ====================================================================
def verifica_pipeline():
    if CP.MODALITA_CLASSI != 6 or CP.NUM_CLASSES != 6:
        raise RuntimeError(
            f"config.py e' in MODALITA_CLASSI={CP.MODALITA_CLASSI}. La "
            f"validazione esterna richiede il modello a 6 classi fini.")
    if list(CP.CLASS_NAMES) != CM.NOMI_PRIVATO:
        raise RuntimeError(
            f"ordine delle classi inatteso: {list(CP.CLASS_NAMES)} != "
            f"{CM.NOMI_PRIVATO}. LUT_PRED sarebbe sbagliata.")
    if CP.TILE != CM.TILE:
        print(f"   ATTENZIONE: TILE privato {CP.TILE} != MTCHI {CM.TILE}")
    print(f"   pipeline verificata: 6 classi, {', '.join(CP.CLASS_NAMES)}")


def transforms_val():
    """
    Gli stessi transform di validazione della pipeline privata. Se dati.py
    non e' importabile si ricostruisce l'equivalente e lo si dichiara a
    schermo, invece di divergere in silenzio.
    """
    try:
        import dati as DP
        return DP.transforms_val(), "dati.transforms_val()"
    except Exception as e:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        print(f"   NOTA: dati.py non importabile ({type(e).__name__}), "
              f"uso A.Normalize()+ToTensorV2")
        return A.Compose([A.Normalize(), ToTensorV2()]), "ricostruito"


# ====================================================================
# 2. DATASET
# ====================================================================
class TileMTCHI(Dataset):
    """
    Un elemento = un tile. La maschera viene restituita in spazio MTCHI
    (0-4): la mappatura a NEGATIVE/LSIL/HSIL avviene fuori, cosi' resta
    verificabile e si puo' cambiare senza ricaricare i dati.
    """

    def __init__(self, voci, transform):
        self.voci = voci
        self.transform = transform

    def __len__(self):
        return len(self.voci)

    def __getitem__(self, i):
        v = self.voci[i]
        img = cv2.imread(v["img"], cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"lettura fallita: {v['img']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        msk = cv2.imread(v["msk"], cv2.IMREAD_GRAYSCALE)
        if msk is None:
            raise RuntimeError(f"lettura fallita: {v['msk']}")
        out = self.transform(image=img)
        return out["image"], torch.from_numpy(msk.astype(np.int64)), i


def elenca_tile(usa_vahadane):
    """
    L'elenco parte SEMPRE dalle maschere: cosi' le quattro configurazioni
    valutano esattamente gli stessi tile e i denominatori sono identici.
    Se manca un'immagine ci si ferma, invece di confrontare medie calcolate
    su insiemi diversi.
    """
    cart_img = CM.CARTELLA_PREPROCESSED if usa_vahadane else CM.CARTELLA_IMG
    p_rep = os.path.join(CM.DIR_USCITA, CM.NOME_REPORT)
    if not os.path.exists(p_rep):
        raise RuntimeError(f"{p_rep} assente: eseguire prima "
                           f"estrai_tile_mtchi.py")
    with open(p_rep, encoding="utf-8") as f:
        rep = json.load(f)
    info = {r["cartella"]: r for r in rep["per_roi"]}

    voci, mancanti = [], []
    for cart in sorted(info):
        base = os.path.join(CM.DIR_USCITA, cart)
        d_img = os.path.join(base, cart_img)
        d_msk = os.path.join(base, CM.CARTELLA_MSK)
        if not os.path.isdir(d_img):
            mancanti.append(cart)
            continue
        for f in sorted(os.listdir(d_msk)):
            p_img = os.path.join(d_img, f)
            if not os.path.exists(p_img):
                mancanti.append(f"{cart}/{f}")
                continue
            voci.append({"img": p_img, "msk": os.path.join(d_msk, f),
                         "tile": f, "cartella": cart,
                         "paziente": info[cart]["paziente"],
                         "partizione": info[cart]["partizione"],
                         "roi": info[cart]["nome"]})
    if mancanti:
        raise RuntimeError(
            f"{len(mancanti)} elementi mancanti in '{cart_img}' "
            f"(es. {mancanti[:3]}). Per le celle 'vahadane_si' occorre prima "
            f"eseguire PreProcess_Dataset.py su {CM.DIR_USCITA} con "
            f"HE_TARGET_FISSA impostato sul report del dataset privato.")
    return voci, rep


# ====================================================================
# 3. MODELLI E INFERENZA
# ====================================================================
def carica_modelli(dir_ckpt, device):
    modelli, nomi = [], []
    for k in range(CM.N_FOLD):
        p = os.path.join(dir_ckpt, f"{CM.PREFISSO_CKPT}{k}.pth")
        if not os.path.exists(p):
            print(f"   fold {k}: checkpoint mancante ({p})")
            continue
        m = MP.build_model(device, verbose=False)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        m.eval()
        modelli.append(m)
        nomi.append(f"fold {k}")
    if not modelli:
        raise RuntimeError(f"nessun checkpoint in {dir_ckpt}")
    print(f"   modelli: {len(modelli)} ({', '.join(nomi)})")
    return modelli, nomi


@torch.no_grad()
def softmax_ensemble(modelli, x, amp, tta, per_fold=False):
    """
    Media delle PROBABILITA', non dei logit: i logit di modelli diversi non
    sono su scale confrontabili, le probabilita' sono tutte normalizzate a 1
    e la media resta una distribuzione valida. Identico a test.py.

    Con per_fold=True restituisce anche l'argmax di ciascun fold. Costa
    pochi MB (uint8) e serve alla diagnostica: dire se l'ensemble e' un
    consenso o e' trainato da un solo fold. NON serve a sceglierne uno.
    """
    varianti = [x]
    if tta:
        varianti += [torch.flip(x, [3]), torch.flip(x, [2]),
                     torch.flip(x, [2, 3])]
    somma, n = None, 0
    singoli = []
    usa_amp = amp and x.is_cuda
    for m in modelli:
        acc_m, n_m = None, 0
        for i, v in enumerate(varianti):
            with torch.amp.autocast("cuda", enabled=usa_amp):
                p = m(v).float().softmax(1)
            if i == 1:
                p = torch.flip(p, [3])
            elif i == 2:
                p = torch.flip(p, [2])
            elif i == 3:
                p = torch.flip(p, [2, 3])
            acc_m = p if acc_m is None else acc_m + p
            n_m += 1
        acc_m = acc_m / n_m
        if per_fold:
            singoli.append(acc_m.argmax(1).cpu().numpy().astype(np.uint8))
        somma = acc_m if somma is None else somma + acc_m
        n += 1
    return (somma / n, singoli) if per_fold else (somma / n, None)


def decidi(prob, omega):
    """-> dict decisione -> tensore (B, H, W) con le 6 classi fini."""
    out = {"argmax": prob.argmax(1)}
    if omega is not None and "costo_minimo" in CM.DECISIONI:
        # y = argmin_j sum_i p_i Omega[i, j]
        rischio = torch.einsum("bchw,cd->bdhw", prob, omega)
        out["costo_minimo"] = rischio.argmin(1)
    return out


# ====================================================================
# 4. ESECUZIONE DI UNA CONFIGURAZIONE
# ====================================================================
def esegui(nome_config, device=None):
    cfg = next((c for c in CM.CONFIGURAZIONI if c["nome"] == nome_config), None)
    if cfg is None:
        raise ValueError(f"configurazione sconosciuta: {nome_config}")

    device = device or torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
    d_out = CM.dir_config(nome_config)
    os.makedirs(d_out, exist_ok=True)

    titolo(f"VALIDAZIONE ESTERNA — {nome_config}")
    print(f"  {CM.firma()}")
    print(f"  Vahadane: {'si' if cfg['vahadane'] else 'no'}   K: {cfg['k']}")
    if cfg["k"] == "k3":
        print("  NOTA: l'intervallo di K=3.0 e' stato calibrato sulle statistiche")
        print("  di densita' ottica di MTCHI (hed_augment.py). Questa cella NON e'")
        print("  una validazione cieca rispetto al dominio target: va riportata")
        print("  come configurazione secondaria.")

    verifica_pipeline()
    transform, origine = transforms_val()
    print(f"   transform: {origine}")
    modelli, nomi_mod = carica_modelli(CM.DIR_MODELLI[cfg["k"]], device)
    omega = (torch.tensor(CP.MATRICE_COSTI, dtype=torch.float32, device=device)
             if getattr(CP, "MATRICE_COSTI", None) is not None else None)
    if omega is None:
        print("   MATRICE_COSTI assente: solo argmax")

    voci, rep_estr = elenca_tile(cfg["vahadane"])
    pazienti = sorted({v["paziente"] for v in voci})
    print(f"   tile: {len(voci)}   RoI: {len({v['cartella'] for v in voci})}"
          f"   PAZIENTI: {len(pazienti)}")

    decisioni = [d for d in CM.DECISIONI if d == "argmax" or omega is not None]
    loader = DataLoader(TileMTCHI(voci, transform), batch_size=CM.BATCH_TEST,
                        shuffle=False, num_workers=CM.NUM_WORKERS,
                        pin_memory=torch.cuda.is_available())

    # accumulatori: per decisione -> per RoI -> matrici
    vuoto = lambda: {"cm": np.zeros((3, 3), np.int64),
                     "cm_esc": np.zeros((3, 3), np.int64),
                     "cm_strat": np.zeros((3, 3), np.int64),
                     "px_vetro": 0, "px_annotati": 0}
    stato = {d: defaultdict(vuoto) for d in decisioni}
    tile_rec = {d: [] for d in decisioni}
    coppie = []                      # per il controllo di equivalenza NLH
    esempi_cand = []

    # Export: si accumula direttamente nelle mappe RoI invece di tenere in RAM
    # migliaia di tile separati. I buchi restano a 0, che con
    # MIN_FRAZIONE_ANNOTAZIONE = 0 coincide con i pixel dove la ground truth e'
    # Background, ignorati comunque da eval_task2.py.
    info_roi = {r["cartella"]: r for r in rep_estr["per_roi"]}
    export = {}

    # Predizioni salvate su disco solo per la configurazione degli esempi:
    # tenerle in RAM costerebbe centinaia di MB, e servono solo per le poche
    # figure selezionate alla fine.
    salva_pred = CM.SALVA_ESEMPI
    dir_pred = os.path.join(d_out, "pred6")
    dir_pred_costo = os.path.join(d_out, "pred6_costo")

    # Diagnostica per fold: matrici di confusione dei singoli modelli, per
    # dire se l'ensemble e' un consenso o poggia su un fold solo.
    cm_fold = [np.zeros((3, 3), np.int64) for _ in modelli]

    d0 = CM.DECISIONE_PRIMARIA
    n_px = CM.TILE * CM.TILE
    fatto = 0

    with torch.no_grad():
        for x, msk, idx in loader:
            x = x.to(device, non_blocking=True)
            prob, singoli = softmax_ensemble(modelli, x, CM.USA_AMP,
                                             CM.USA_TTA, per_fold=True)
            pred = {d: v.cpu().numpy().astype(np.uint8)
                    for d, v in decidi(prob, omega).items()}
            msk = msk.numpy().astype(np.uint8)

            for b in range(msk.shape[0]):
                v = voci[int(idx[b])]
                t3 = ME.mappa_truth(msk[b])
                valido = t3 != CM.IGNORE
                n_ann = int(valido.sum())
                copertura = n_ann / n_px

                for kf, sing in enumerate(singoli):
                    pf, _ = ME.mappa_predizione(sing[b], "negativo")
                    cm_fold[kf] += ME.confusione(pf, t3)

                for d in decisioni:
                    p6 = pred[d][b]
                    p_neg, vetro = ME.mappa_predizione(p6, "negativo")
                    p_esc, _ = ME.mappa_predizione(p6, "escluso")
                    cm = ME.confusione(p_neg, t3)
                    s = stato[d][v["cartella"]]
                    s["cm"] += cm
                    s["cm_esc"] += ME.confusione(p_esc, t3)
                    s["px_vetro"] += int((vetro & valido).sum())
                    s["px_annotati"] += n_ann
                    if copertura >= CM.SOGLIA_STRATIFICAZIONE:
                        s["cm_strat"] += cm

                    # rilevazione per tile: quota di pixel della lesione
                    # predetta POSITIVA (LSIL o HSIL) e quota col grado esatto
                    pos = (p_neg == CM.IDX_LSIL) | (p_neg == CM.IDX_HSIL)
                    n_vera = np.array([int((t3 == c).sum()) for c in range(3)])
                    tile_rec[d].append({
                        "paziente": v["paziente"],
                        "cartella": v["cartella"], "tile": v["tile"],
                        "copertura": float(copertura),
                        "n_vera": n_vera,
                        "n_positivo": np.array(
                            [int(((t3 == c) & pos).sum()) for c in range(3)]),
                        "n_esatto": np.array(
                            [int(((t3 == c) & (p_neg == c)).sum())
                             for c in range(3)]),
                    })

                    if d == d0:
                        coppie.append((p_neg, t3))
                        if CM.ESPORTA_DAT:
                            cart = v["cartella"]
                            if cart not in export:
                                W, H = info_roi[cart]["dimensione"]
                                export[cart] = np.zeros((H, W), dtype=np.uint8)
                            out = np.array(CM.LUT_INV_MTCHI,
                                           dtype=np.uint8)[p6]
                            out[msk[b] == 0] = 0   # come il background_filter
                            y = int(v["tile"][1:7])
                            x = int(v["tile"][9:15])
                            M = export[cart]
                            y2 = min(y + CM.TILE, M.shape[0])
                            x2 = min(x + CM.TILE, M.shape[1])
                            M[y:y2, x:x2] = out[:y2 - y, :x2 - x]
                        if salva_pred:
                            dp = os.path.join(dir_pred, v["cartella"])
                            os.makedirs(dp, exist_ok=True)
                            cv2.imwrite(os.path.join(dp, v["tile"]), p6)
                            esempi_cand.append(
                                FG.candidato(v, cm, t3, p_neg, vetro, n_ann))
                    elif d == "costo_minimo" and salva_pred:
                        dp = os.path.join(dir_pred_costo, v["cartella"])
                        os.makedirs(dp, exist_ok=True)
                        cv2.imwrite(os.path.join(dp, v["tile"]), p6)

            fatto += msk.shape[0]
            if fatto % (CM.BATCH_TEST * 20) < CM.BATCH_TEST:
                print(f"    {fatto}/{len(voci)} tile")

    # ---------------- metriche ----------------
    report = {"configurazione": cfg, "firma": CM.firma(), "modelli": nomi_mod,
              "n_tile": len(voci), "n_roi": len(stato[d0]),
              "n_pazienti": len(pazienti), "pazienti": pazienti,
              "politica_vetro": CM.POLITICA_VETRO, "decisioni": {}}
    righe = []

    for d in decisioni:
        per_roi = {}
        for cart, s in stato[d].items():
            info = next(v for v in voci if v["cartella"] == cart)
            per_roi[cart] = {
                "paziente": info["paziente"],
                "partizione": info["partizione"],
                "roi": info["roi"],
                "cm": s["cm"].tolist(),
                "cm_escluso_vetro": s["cm_esc"].tolist(),
                "cm_stratificata": s["cm_strat"].tolist(),
                "miou_presenti": ME.macro_presenti(s["cm"]),
                "quota_predetta_vetro": (s["px_vetro"] /
                                         max(s["px_annotati"], 1)),
                "pixel_annotati": int(s["px_annotati"]),
                "sens_hsil": (float(s["cm"][2, 2] / s["cm"][2].sum())
                              if s["cm"][2].sum() > 0 else None),
                "lsil_e_hsil_insieme": bool(s["cm"][1].sum() > 0
                                            and s["cm"][2].sum() > 0),
            }

        cm = np.sum([np.array(v["cm"]) for v in per_roi.values()], axis=0)
        cm_esc = np.sum([np.array(v["cm_escluso_vetro"])
                         for v in per_roi.values()], axis=0)
        cm_str = np.sum([np.array(v["cm_stratificata"])
                         for v in per_roi.values()], axis=0)
        q_vetro = float(np.mean([v["quota_predetta_vetro"]
                                 for v in per_roi.values()]))
        ris = ME.riassunto(cm, quota_vetro=q_vetro, cm_escluso=cm_esc)

        ic = {
            "sens_hsil": ME.bootstrap_aggregato(
                per_roi, lambda m: ME.binario_hsil(m)["sensibilita"]),
            "spec_hsil": ME.bootstrap_aggregato(
                per_roi, lambda m: ME.binario_hsil(m)["specificita"]),
            "spec_tessuto": ME.bootstrap_aggregato(
                per_roi, lambda m: ME.binario_hsil(m)["specificita"],
                campo="cm_escluso_vetro"),
            "miou": ME.bootstrap_aggregato(per_roi, ME.macro_presenti),
        }
        ris["ic95_bootstrap_paziente"] = {k: list(v) for k, v in ic.items()}
        ris["jackknife_sens_hsil"] = ME.jackknife_paziente(
            per_roi, lambda m: ME.binario_hsil(m)["sensibilita"])
        ris["rilevazione"] = ME.rilevazione_per_tile(tile_rec[d])

        # --- stratificazione per copertura ---------------------------
        # NON e' un filtro: le metriche principali usano tutti i tile.
        # Risponde a "il risultato dipende dai tile di bordo?". Se le due
        # righe coincidono, il risultato non dipende dalla giunzione
        # epitelio-stroma. Se divergono, e' una misura di quanto il modello
        # soffra proprio li', ed e' un'osservazione, non un difetto.
        b_str = ME.binario_hsil(cm_str)
        ris["stratificazione_copertura"] = {
            "soglia": CM.SOGLIA_STRATIFICAZIONE,
            "tutti": {"sens_hsil": ris["hsil_vs_non_hsil"]["sensibilita"],
                      "spec_hsil": ris["hsil_vs_non_hsil"]["specificita"],
                      "miou": ris["miou_3classi"],
                      "px": int(cm.sum())},
            "solo_alta_copertura": {"sens_hsil": float(b_str["sensibilita"]),
                                    "spec_hsil": float(b_str["specificita"]),
                                    "miou": ME.macro_presenti(cm_str),
                                    "px": int(cm_str.sum())},
        }

        # --- stratificazione per partizione ufficiale -----------------
        parti = sorted({v["partizione"] for v in per_roi.values()})
        ris["per_partizione"] = {}
        if len(parti) > 1:
            for part in parti:
                sub = {k: v for k, v in per_roi.items()
                       if v["partizione"] == part}
                cs = np.sum([np.array(v["cm"]) for v in sub.values()], axis=0)
                bs = ME.binario_hsil(cs)
                ris["per_partizione"][part] = {
                    "n_roi": len(sub),
                    "n_pazienti": len({v["paziente"] for v in sub.values()}),
                    "sens_hsil": float(bs["sensibilita"]),
                    "spec_hsil": float(bs["specificita"]),
                    "miou_3classi": ME.macro_presenti(cs),
                }

        v_roi = [v["miou_presenti"] for v in per_roi.values()
                 if v["miou_presenti"] is not None
                 and np.isfinite(v["miou_presenti"])]
        ris["miou_media_per_roi"] = float(np.mean(v_roi)) if v_roi else None

        ris["cm"] = cm.tolist()
        ris["cm_escluso_vetro"] = cm_esc.tolist()
        ris["per_roi"] = per_roi
        report["decisioni"][d] = ris
        righe.append(FG.riga_tabella(nome_config, d, report, ris, ic,
                                     q_vetro, cm))

    # ---------------- controllo di equivalenza ----------------
    uff = ME.nlh_miou_ufficiale(coppie)
    nostro = report["decisioni"][d0]["miou_3classi"]
    report["controllo_equivalenza_nlh"] = {"ufficiale": uff, "nostro": nostro,
                                           "delta": abs(uff - nostro)}
    titolo("CONTROLLO DI EQUIVALENZA CON eval_task2.py (MCPRL)")
    print(f"  mIoU da matrice di confusione      : {nostro:.6f}")
    print(f"  mIoU con l'algoritmo NLH ufficiale : {uff:.6f}")
    print(f"  differenza                         : {abs(uff-nostro):.2e}")
    print("  Identiche: la mIoU riportata E' la NLH-mIoU ufficiale."
          if abs(uff - nostro) < 1e-9 else
          "  DIVERGENZA: errore nella conversione o nell'aggregazione.\n"
          "  Non usare questi numeri finche' non e' risolta.")

    sens_fold = {f"fold {i}": (float(c[2, 2] / c[2].sum())
                               if c[2].sum() > 0 else None)
                 for i, c in enumerate(cm_fold)}
    report["sens_hsil_per_fold"] = sens_fold
    FG.stampa_per_fold(
        sens_fold,
        report["decisioni"][d0]["hsil_vs_non_hsil"]["sensibilita"])

    if len(pazienti) < CM.N_MINIMO_PER_IC:
        print(f"\n  ATTENZIONE: {len(pazienti)} pazienti. Con questo n il")
        print("  bootstrap e' esso stesso instabile: riportare l'intervallo,")
        print("  mai la sola stima puntuale, e non usare questo insieme come")
        print("  test di superiorita' fra varianti.")

    # ---------------- uscite ----------------
    FG.stampa_tabella(righe)
    FG.stampa_rilevazione(report["decisioni"][d0]["rilevazione"])
    FG.stampa_stratificazioni(report["decisioni"][d0])

    with open(os.path.join(d_out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)
    FG.salva_tabella(righe, os.path.join(d_out, "tabella.csv"))
    FG.plot_confusione(report["decisioni"][d0]["cm"],
                       report["decisioni"][d0]["hsil_vs_non_hsil"],
                       os.path.join(d_out, "confusione.png"),
                       f"MTCHI Task 2 — {nome_config} ({d0})")

    if CM.ESPORTA_DAT and export:
        FG.esporta_dat(export, rep_estr,
                       os.path.join(d_out, "predizioni.dat"))

    if salva_pred:
        FG.salva_esempi(esempi_cand, report["decisioni"][d0]["cm"],
                        os.path.join(d_out, "esempi"), dir_pred,
                        dir_pred_costo if os.path.isdir(dir_pred_costo)
                        else None)
        for d in (dir_pred, dir_pred_costo):
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)

    print(f"\n  Uscite in: {d_out}")
    return report, righe


# ====================================================================
# 5. MAIN
# ====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configurazione", nargs="?", default=None)
    args = ap.parse_args()

    os.makedirs(CM.DIR_RISULTATI, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    nomi = ([args.configurazione] if args.configurazione
            else [c["nome"] for c in CM.CONFIGURAZIONI])
    tutte, riepilogo, saltate = [], {}, []

    for n in nomi:
        try:
            rep, righe = esegui(n, device)
        except RuntimeError as e:
            print(f"\n  CONFIGURAZIONE '{n}' SALTATA: {e}\n")
            saltate.append((n, str(e)))
            continue
        tutte.extend(righe)
        d = rep["decisioni"][CM.DECISIONE_PRIMARIA]
        ic = d["ic95_bootstrap_paziente"]
        riepilogo[n] = {
            "sens_hsil": d["hsil_vs_non_hsil"]["sensibilita"],
            "spec_tessuto": d.get("specificita_su_tessuto", float("nan")),
            "miou": d["miou_3classi"],
            "recall_lsil": d["recall_per_classe"][CM.IDX_LSIL],
            "quota_vetro": d.get("quota_predetta_vetro", float("nan")),
            "ic": {k: tuple(v) for k, v in ic.items()},
        }

    if len(riepilogo) > 1:
        FG.confronto(riepilogo, tutte, saltate)
    if saltate:
        print(f"\n  {len(saltate)} configurazioni saltate:")
        for n, m in saltate:
            print(f"    {n}: {m[:110]}")


if __name__ == "__main__":
    sys.exit(main())