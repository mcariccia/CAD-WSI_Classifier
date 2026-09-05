"""
extraction_tiles.py
====================================================================
Estrae tile 512x512 dalle RoI di MTCHI Task 2, nella stessa struttura
di Dataset_Tiles del dataset privato.

    <DIR_USCITA>/<nome_RoI>/images/y000000_x000512.png
    <DIR_USCITA>/<nome_RoI>/masks/y000000_x000512.png

COSA CAMBIA RISPETTO A extraction_tiles.py
-------------------------------------------
1. UNA CARTELLA PER RoI, non tutto insieme. PreProcess_Dataset.py stima una
   matrice di stain per sottocartella: con una cartella piatta ne stimerebbe
   una sola per venti vetrini diversi. Serve anche a raggruppare per paziente
   nel bootstrap.

2. OVERLAP 0 invece di 0.75. Con stride 128 ogni pixel finiva in decine di
   tile: su un training set e' augmentation, su un test set e' pseudo-
   replicazione che gonfia i conteggi e falsifica gli intervalli.

3. VIA IL FILTRO SULLA MASCHERA. Scartare i tile con meno del 10% di
   annotazione e' selezione sull'ETICHETTA: si sceglie dove essere valutati e
   il risultato e' ottimistico per costruzione. Resta solo il filtro sul nero,
   che agisce sull'IMMAGINE ed elimina un artefatto del ritaglio poligonale,
   non tessuto difficile.

4. NERO -> BIANCO nei tile conservati. In H&E il vuoto e' vetro; il nero ha
   densita' ottica massima e verrebbe letto come ematossilina intensissima.
   Prima di Vahadane, altrimenti domina la stima NMF.

5. TILE DI BORDO ALLINEATI al margine invece che scartati, cosi' la griglia
   copre la RoI intera.

6. NOMI ORDINABILI riga-per-riga (y000000_x000512): raccogli_pixel_wsi di
   PreProcess_Dataset campiona con np.linspace sui nomi ordinati, e con nomi
   non allineati campionerebbe un angolo solo del vetrino.

Le etichette restano quelle originali (0=BG, 1=Normal, 2=CIN1, 3=CIN2, 4=CIN3):
la mappatura a NEGATIVE/LSIL/HSIL si fa in fase di test, cosi' si puo'
cambiare senza riestrarre.

DOPO
-----
    PreProcess_Dataset.py con
        DATASET_DIR     = DIR_USCITA
        HE_TARGET_FISSA = report_normalizzazione.json del dataset PRIVATO
    -> crea images_preprocessed/ in ogni cartella
====================================================================
"""

import json
import os
import re
import sys

import cv2
import numpy as np

import config_esterno as CM

RE_PAZIENTE = re.compile(r"Crop_(B\d+)[-_]", re.IGNORECASE)


def riga(c="-", n=78):
    print(c * n)


def paziente_da_nome(nome):
    m = RE_PAZIENTE.search(nome)
    return m.group(1) if m else os.path.splitext(nome)[0]


# ====================================================================
# ELENCO
# ====================================================================
def elenca_roi():
    voci, visti = [], {}
    for part in CM.PARTIZIONI:
        d_img = os.path.join(CM.DIR_MTCHI, part, "Img")
        d_msk = os.path.join(CM.DIR_MTCHI, part, "Truth")
        if not os.path.isdir(d_img):
            print(f"  ATTENZIONE: partizione '{part}' assente ({d_img})")
            continue
        for f in sorted(os.listdir(d_img)):
            if not f.lower().endswith(CM.ESTENSIONI):
                continue
            p_msk = os.path.join(d_msk, "Mask_" + f)
            if not os.path.exists(p_msk):
                print(f"  {part}/{f}: maschera assente, RoI saltata")
                continue
            if f in visti:
                raise RuntimeError(f"nome duplicato fra partizioni: '{f}' in "
                                   f"{visti[f]} e in {part}")
            visti[f] = part
            voci.append({"nome": f, "partizione": part,
                         "img": os.path.join(d_img, f), "truth": p_msk,
                         "paziente": paziente_da_nome(f)})
    if not voci:
        raise RuntimeError(f"nessuna RoI trovata sotto {CM.DIR_MTCHI}")
    return voci


# ====================================================================
# GRIGLIA E NERO
# ====================================================================
def posizioni(dim, tile, stride, allinea):
    """Coordinate di partenza lungo un asse."""
    if dim <= tile:
        return [0]
    p = list(range(0, dim - tile + 1, stride))
    if allinea and p[-1] != dim - tile:
        p.append(dim - tile)
    return p


def maschera_nero(rgb):
    """
    Fondo fuori RoI: nero pieno, in componenti connesse GRANDI.

    Il filtro per area e' la protezione vera. Senza, la dilatazione
    allargherebbe di qualche pixel anche il nucleo ipercromatico piu' scuro
    e ne imbiancherebbe il contorno. Con AREA_MINIMA_NERO il nucleo (100-300
    px) non entra mai nella maschera, mentre il fondo del ritaglio (milioni
    di px) resta intero.
    """
    nero = np.all(rgb <= CM.SOGLIA_NERO, axis=2).astype(np.uint8)
    if not nero.any():
        return nero.astype(bool)

    if CM.AREA_MINIMA_NERO > 0:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(nero, connectivity=8)
        tieni = np.zeros(n, dtype=bool)
        for i in range(1, n):
            tieni[i] = stats[i, cv2.CC_STAT_AREA] >= CM.AREA_MINIMA_NERO
        nero = tieni[lab].astype(np.uint8)
        if not nero.any():
            return nero.astype(bool)

    if CM.DILATA_NERO > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * CM.DILATA_NERO + 1,) * 2)
        nero = cv2.dilate(nero, k, iterations=1)
    return nero.astype(bool)


# ====================================================================
# ESTRAZIONE
# ====================================================================
def estrai_roi(voce):
    nome = voce["nome"]
    img = cv2.imread(voce["img"], cv2.IMREAD_COLOR)      # BGR
    msk = cv2.imread(voce["truth"], cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        raise RuntimeError(f"{nome}: lettura fallita")
    if img.shape[:2] != msk.shape[:2]:
        raise RuntimeError(f"{nome}: immagine {img.shape[:2]} e maschera "
                           f"{msk.shape[:2]} non coincidono")
    fuori = set(np.unique(msk).tolist()) - CM.VALORI_AMMESSI
    if fuori:
        raise RuntimeError(f"{nome}: valori {sorted(fuori)} non previsti. "
                           f"Attesi 0=BG, 1=Normal, 2=CIN1, 3=CIN2, 4=CIN3.")

    H, W = img.shape[:2]
    T = CM.TILE
    stride = max(1, int(round(T * (1.0 - CM.OVERLAP))))
    nero = maschera_nero(img)

    base = CM.cartella_roi(voce["cartella"])
    d_img = os.path.join(base, CM.CARTELLA_IMG)
    d_msk = os.path.join(base, CM.CARTELLA_MSK)
    os.makedirs(d_img, exist_ok=True)
    os.makedirs(d_msk, exist_ok=True)

    ys = posizioni(H, T, stride, CM.ALLINEA_BORDI)
    xs = posizioni(W, T, stride, CM.ALLINEA_BORDI)

    n_griglia = len(ys) * len(xs)
    n_scartati = n_salvati = 0
    px_nero_sost = 0
    ann_conservati = np.zeros(5, dtype=np.int64)
    assegnati = np.zeros((H, W), dtype=bool)
    min_px = CM.MIN_FRAZIONE_ANNOTAZIONE * T * T

    for y in ys:
        for x in xs:
            y2, x2 = min(y + T, H), min(x + T, W)
            t_img = img[y:y2, x:x2]
            t_msk = msk[y:y2, x:x2]

            # RoI piu' piccola del tile: si completa con bianco, mai con nero
            if t_img.shape[0] != T or t_img.shape[1] != T:
                pad = np.full((T, T, 3), CM.VALORE_BIANCO, dtype=np.uint8)
                pad[:t_img.shape[0], :t_img.shape[1]] = t_img
                t_img = pad
                padm = np.zeros((T, T), dtype=np.uint8)
                padm[:t_msk.shape[0], :t_msk.shape[1]] = t_msk
                t_msk = padm
                t_nero = np.zeros((T, T), dtype=bool)
                t_nero[:y2 - y, :x2 - x] = nero[y:y2, x:x2]
            else:
                t_msk = t_msk.copy()
                t_nero = nero[y:y2, x:x2]

            # --- niente pixel contati due volte -------------------------
            if CM.EVITA_DOPPIO_CONTEGGIO:
                gia = assegnati[y:y2, x:x2]
                if gia.any():
                    t_msk[:y2 - y, :x2 - x][gia] = 0

            # --- criterio di accettazione -------------------------------
            n_ann = int(np.count_nonzero(t_msk))
            if n_ann == 0 or n_ann < min_px:
                n_scartati += 1
                continue

            if CM.SOSTITUISCI_NERO and t_nero.any():
                t_img = t_img.copy()
                t_img[t_nero] = CM.VALORE_BIANCO
                px_nero_sost += int(t_nero.sum())

            f = f"y{y:06d}_x{x:06d}.png"
            cv2.imwrite(os.path.join(d_img, f), t_img)
            cv2.imwrite(os.path.join(d_msk, f), t_msk)
            n_salvati += 1
            assegnati[y:y2, x:x2] = True
            ann_conservati += np.bincount(t_msk.ravel(), minlength=5)[:5]

    ann_roi = np.bincount(msk.ravel(), minlength=5)[:5].astype(np.int64)
    tot_roi = int(ann_roi[1:].sum())
    tot_tile = int(ann_conservati[1:].sum())
    copertura = tot_tile / tot_roi if tot_roi else 0.0

    return {
        "nome": nome,
        "cartella": voce["cartella"],
        "partizione": voce["partizione"],
        "paziente": voce["paziente"],
        "dimensione": [int(W), int(H)],
        "tile_griglia": n_griglia,
        "tile_scartati": n_scartati,
        "tile_salvati": n_salvati,
        "px_nero_sostituiti": int(px_nero_sost),
        "px_nero_roi": int(nero.sum()),
        "pixel_per_classe_roi": ann_roi.tolist(),
        "pixel_per_classe_tile": ann_conservati.tolist(),
        "copertura_annotazioni": float(copertura),
    }


def sonda(voci, soglie=(0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)):
    """
    Quanta annotazione sopravvive a ciascuna soglia MIN_FRAZIONE_ANNOTAZIONE,
    senza scrivere niente.

    La soglia va scelta QUI, guardando il costo, e non dopo aver visto le
    prestazioni del modello: sceglierla per far salire una metrica sarebbe
    adattamento al test.

    Da leggere insieme alla colonna 'px persi': quelli non sono pixel
    qualunque. Un tile con poca annotazione sta quasi sempre sul BORDO di un
    frammento epiteliale, alla giunzione con lo stroma dove si trova lo
    strato basale e dove si applica il criterio LAST. Sono i pixel piu'
    difficili, quindi il guadagno di metrica che si ottiene alzando la soglia
    non e' una misura del modello.
    """
    print("\nSONDA — costo della soglia di annotazione (nessun file scritto)\n")

    T, stride = CM.TILE, max(1, int(round(CM.TILE * (1.0 - CM.OVERLAP))))
    acc = {s: {"tile": 0, "px": np.zeros(5, dtype=np.int64)} for s in soglie}
    tot = np.zeros(5, dtype=np.int64)
    n_griglia = 0

    for v in voci:
        img = cv2.imread(v["img"], cv2.IMREAD_COLOR)
        msk = cv2.imread(v["truth"], cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue
        H, W = img.shape[:2]
        tot += np.bincount(msk.ravel(), minlength=5)[:5]
        # una maschera di "gia' assegnato" per soglia: l'insieme dei tile
        # conservati cambia con la soglia, quindi cambia anche quale tile di
        # bordo rivendica i pixel sovrapposti
        vista = {s: np.zeros((H, W), dtype=bool) for s in soglie}

        for y in posizioni(H, T, stride, CM.ALLINEA_BORDI):
            for x in posizioni(W, T, stride, CM.ALLINEA_BORDI):
                y2, x2 = min(y + T, H), min(x + T, W)
                sub = msk[y:y2, x:x2]
                n_griglia += 1
                for s in soglie:
                    m = sub
                    if CM.EVITA_DOPPIO_CONTEGGIO:
                        m = np.where(vista[s][y:y2, x:x2], 0, sub)
                    n_ann = int(np.count_nonzero(m))
                    if n_ann == 0 or n_ann < s * T * T:
                        continue
                    if CM.EVITA_DOPPIO_CONTEGGIO:
                        vista[s][y:y2, x:x2] = True
                    acc[s]["tile"] += 1
                    acc[s]["px"] += np.bincount(m.ravel(), minlength=5)[:5]

    base = max(int(tot[1:].sum()), 1)
    print(f"  {'soglia':>8}{'tile':>9}{'annotaz. conservata':>23}{'px persi':>14}")
    riga(".")
    for s in soglie:
        cop = acc[s]["px"][1:].sum() / base
        persi = base - int(acc[s]["px"][1:].sum())
        nota = "   <-- nessuna selezione" if s == 0.0 else ""
        print(f"  {100*s:>7.0f}%{acc[s]['tile']:>9}{100*cop:>22.1f}%"
              f"{persi:>14,}{nota}")
    riga(".")

    print(f"\n  {'soglia':>8}   annotazione conservata per classe")
    riga(".")
    for s in soglie:
        q = []
        for k in range(1, 5):
            q.append(f"{CM.NOMI_MTCHI[k]} "
                     f"{100*acc[s]['px'][k]/max(tot[k],1):>5.1f}%"
                     if tot[k] else f"{CM.NOMI_MTCHI[k]}     -")
        print(f"  {100*s:>7.0f}%   " + "   ".join(q))
    riga(".")
    print("\n  Guardare se le classi perdono in modo DIVERSO: un frammento")
    print("  compatto sopravvive, una striscia sottile no. Se la perdita non e'")
    print("  uniforme, la soglia non sta solo scartando tile, sta cambiando la")
    print("  composizione del test set.")
    print(f"\n  Scrivere la soglia scelta in MIN_FRAZIONE_ANNOTAZIONE "
          f"(ora {CM.MIN_FRAZIONE_ANNOTAZIONE}).")


# ====================================================================
# MAIN
# ====================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sonda", action="store_true",
                    help="misura la copertura a varie soglie senza scrivere")
    args = ap.parse_args()

    print("Estrazione tile da MTCHI Task 2")
    print(f"  partizioni  : {', '.join(CM.PARTIZIONI)}")
    print(f"  tile        : {CM.TILE}, overlap {CM.OVERLAP}")
    print(f"  nero        : soglia {CM.SOGLIA_NERO}, area min "
          f"{CM.AREA_MINIMA_NERO}, dilatazione {CM.DILATA_NERO} -> bianco")
    print(f"  accetta tile: annotazione >= "
          f"{100*CM.MIN_FRAZIONE_ANNOTAZIONE:.0f}% "
          f"({'almeno 1 px' if CM.MIN_FRAZIONE_ANNOTAZIONE == 0 else 'selezione attiva'})")
    print(f"  uscita      : {CM.DIR_USCITA}")
    print()

    voci = elenca_roi()
    if args.sonda:
        sonda(voci)
        return

    os.makedirs(CM.DIR_USCITA, exist_ok=True)
    for i, v in enumerate(voci, 1):
        v["cartella"] = CM.nome_cartella(v["nome"], i)
    if len({v["cartella"] for v in voci}) != len(voci):
        raise RuntimeError("nomi di cartella duplicati")
    report = []

    intest = (f"  {'cartella':<32}{'tile':>7}{'scart.':>8}{'salvati':>9}"
              f"{'copert.':>9}  classi")
    print(intest)
    riga(".")
    for i, v in enumerate(voci, 1):
        r = estrai_roi(v)
        report.append(r)
        cl = ",".join(CM.NOMI_MTCHI[k] for k in range(1, 5)
                      if r["pixel_per_classe_tile"][k] > 0) or "-"
        print(f"  {r['cartella'][:31]:<32}{r['tile_griglia']:>7}"
              f"{r['tile_scartati']:>8}"
              f"{r['tile_salvati']:>9}"
              f"{100*r['copertura_annotazioni']:>8.1f}%  {cl}")
    riga(".")

    # ---------------- riepilogo ----------------
    tot_tile = sum(r["tile_salvati"] for r in report)
    pazienti = sorted({r["paziente"] for r in report})
    px_roi = np.sum([r["pixel_per_classe_roi"] for r in report], axis=0)
    px_tile = np.sum([r["pixel_per_classe_tile"] for r in report], axis=0)

    print()
    print(f"  RoI       : {len(report)}")
    print(f"  PAZIENTI  : {len(pazienti)}   <-- il vero n dell'esperimento")
    print(f"  Tile      : {tot_tile:,}")
    print()
    print(f"  {'classe':<12}{'px nella RoI':>16}{'px nei tile':>16}{'conservato':>13}")
    riga(".")
    for k in range(1, 5):
        c = px_tile[k] / px_roi[k] if px_roi[k] else 0.0
        print(f"  {CM.NOMI_MTCHI[k]:<12}{px_roi[k]:>16,}{px_tile[k]:>16,}"
              f"{100*c:>12.1f}%")
    riga(".")
    tot_c = px_tile[1:].sum() / max(px_roi[1:].sum(), 1)
    print(f"  {'TOTALE':<12}{px_roi[1:].sum():>16,}{px_tile[1:].sum():>16,}"
          f"{100*tot_c:>12.1f}%")

    vuote = [r["nome"] for r in report if r["tile_salvati"] == 0]
    if vuote:
        print()
        print(f"  ATTENZIONE: {len(vuote)} RoI senza alcun tile salvato:")
        for n in vuote:
            print(f"    {n}")
        print("  Spariscono dal test e cambiano il denominatore. Abbassare")
        print("  MIN_FRAZIONE_ANNOTAZIONE oppure escluderle e dichiararlo.")

    print()
    print("  La colonna 'conservato' e' la quota di annotazione che sopravvive")
    print("  al filtro sul nero. Va riportata in tesi: e' l'unico effetto")
    print("  collaterale del filtro, e dichiararlo lo rende una scelta")
    print("  documentata invece di una perdita invisibile.")
    if CM.MIN_FRAZIONE_ANNOTAZIONE > 0 and tot_c < 0.99:
        print()
        print(f"  ATTENZIONE: la soglia di annotazione al "
              f"{100*CM.MIN_FRAZIONE_ANNOTAZIONE:.0f}% scarta il "
              f"{100*(1-tot_c):.1f}% dei pixel annotati.")
        print("  Non sono pixel qualunque: un tile con poca annotazione sta")
        print("  quasi sempre sul bordo di un frammento epiteliale, alla")
        print("  giunzione con lo stroma dove si applica il criterio LAST.")
        print("  Con MIN_FRAZIONE_ANNOTAZIONE = 0.0 la copertura e' 100%.")

    p = os.path.join(CM.DIR_USCITA, CM.NOME_REPORT)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"parametri": {
            "tile": CM.TILE, "overlap": CM.OVERLAP,
            "soglia_nero": CM.SOGLIA_NERO,
            "area_minima_nero": CM.AREA_MINIMA_NERO,
            "dilata_nero": CM.DILATA_NERO,
            "sostituisci_nero": CM.SOSTITUISCI_NERO,
            "min_frazione_annotazione": CM.MIN_FRAZIONE_ANNOTAZIONE,
            "evita_doppio_conteggio": CM.EVITA_DOPPIO_CONTEGGIO,
            "schema_cartelle": CM.SCHEMA_CARTELLE,
            "partizioni": CM.PARTIZIONI},
            "n_roi": len(report), "n_pazienti": len(pazienti),
            "pazienti": pazienti, "n_tile": tot_tile,
            "copertura_totale": float(tot_c),
            "per_roi": report}, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {p}")
    print()
    print("  PASSO SUCCESSIVO — PreProcess_Dataset.py con:")
    print(f"    DATASET_DIR     = r\"{CM.DIR_USCITA}\"")
    print("    HE_TARGET_FISSA = report_normalizzazione.json del PRIVATO")


if __name__ == "__main__":
    sys.exit(main())