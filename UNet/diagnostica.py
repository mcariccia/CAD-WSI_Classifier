"""
diagnostica.py
====================================================================
Verifiche da eseguire PRIMA del training, e analisi strutturale del
dataset.

Filosofia: fallire subito e rumorosamente. Gli errori che hanno fatto
perdere piu' tempo in questo progetto erano tutti SILENZIOSI —
parametri di augmentation scartati con un warning, maschere lette come
palette invece che come indici, patch applicata a una cartella diversa
da quella letta dal training. Ognuno di essi lasciava girare il
training fino in fondo producendo numeri plausibili e sbagliati.

L'analisi del dataset serve a fissare l'ASPETTATIVA prima di guardare
le metriche: con classi presenti in poche WSI, un numero alto non
significa quello che sembra.
====================================================================
"""

import os
import json
import warnings


import numpy as np
from PIL import Image
import albumentations as A

import config as C
import dati as D


# ====================================================================
# 1. VERIFICHE DI CORRETTEZZA
# ====================================================================
def verifica_riempimento():
    """
    Due controlli distinti.

    (1) DETERMINISTICO: rotate=45 e scale=1.0 garantiscono angoli vuoti,
        quindi se il riempimento e' attivo DEVE comparire il valore di
        fill, e i pixel bianchi dell'immagine devono corrispondere a
        etichette "vetro" nella maschera (coerenza semantica).
        Non si puo' usare affine() per questo: ha p=0.5 e scale fino a
        1.15, e quando la scala supera 1.0 l'immagine viene ingrandita e
        non si formano affatto zone vuote.

    (2) STATISTICO: molte passate della pipeline reale, verificando che
        non nascano etichette fuori dall'insieme atteso (catturerebbe
        un'interpolazione non-nearest sulla maschera).

    In piu' si intercettano i warning "not valid for transform": da
    albumentations 2.x i parametri sbagliati non sollevano eccezioni.
    """
    kw = D.kwargs_riempimento()

    # (1) Verifica del padding geometrico.
    # Inizializziamo una maschera costante su una classe attiva (es. HSIL) e un'immagine scura.
    # Questo garantisce che i nuovi pixel inseriti dalla rotazione siano inequivocabilmente il nostro fill.
    img = np.full((128, 128, 3), 60, dtype=np.uint8)       # Valore di contrasto rispetto al fill atteso
    msk = np.full((128, 128), 3, dtype=np.uint8)           # Maschera omogenea (Classe 3: HSIL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = A.Affine(rotate=(45, 45), scale=(1.0, 1.0), p=1.0, **kw)(
            image=img, mask=msk)

    v_msk = set(np.unique(out["mask"]).tolist())
    v_img = set(np.unique(out["image"]).tolist())
    if D.VALORE_FILL_MSK not in v_msk:
        raise RuntimeError(
            f"Riempimento della maschera non applicato: valori {sorted(v_msk)}, "
            f"atteso {D.VALORE_FILL_MSK} negli angoli introdotti dalla "
            f"rotazione.")
    if D.VALORE_FILL_IMG not in v_img:
        raise RuntimeError(
            f"Riempimento dell'IMMAGINE non applicato: valori "
            f"{sorted(v_img)[:6]}..., atteso {D.VALORE_FILL_IMG}. Col default "
            f"il padding sarebbe nero, che in H&E non esiste.")

     # (2) Immagine e maschera devono dire la stessa cosa.
    # L'augmentation riempie di bianco le zone vuote create dalle
    # rotazioni. Il bianco e' vetro, quindi la maschera li deve dire
    # "Sfondo". Qui si controlla che lo faccia davvero.
    bianchi = (out["image"] == D.VALORE_FILL_IMG).all(axis=2)
    if bianchi.any():
        etichette_bianchi = set(np.unique(out["mask"][bianchi]).tolist())
        if etichette_bianchi != {D.VALORE_FILL_MSK}:
            raise RuntimeError(
                f"Incoerenza fra riempimento immagine e maschera: i pixel "
                f"bianchi hanno etichette {sorted(etichette_bianchi)}, atteso "
                f"solo {D.VALORE_FILL_MSK} ({C.CLASS_NAMES[D.VALORE_FILL_MSK]}).")

    img2 = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    msk2 = np.random.randint(0, C.NUM_CLASSES, (256, 256)).astype(np.uint8)
    tf = A.Compose(D.lista_augmentation())
    visti = set()
    with warnings.catch_warnings(record=True) as catturati:
        warnings.simplefilter("always")
        for _ in range(60):
            visti.update(np.unique(tf(image=img2, mask=msk2)["mask"]).tolist())

    spurie = visti - (set(range(C.NUM_CLASSES)) | {C.IGNORE_INDEX})
    if spurie:
        raise RuntimeError(f"Etichette spurie dopo l'augmentation: "
                           f"{sorted(spurie)}. Probabile interpolazione "
                           f"non-nearest sulla maschera.")
    scartati = [c for c in catturati if "not valid for transform" in str(c.message)]
    if scartati:
        raise RuntimeError("Parametri scartati silenziosamente da "
                           "albumentations:\n  " +
                           "\n  ".join(str(c.message) for c in scartati))

    print(f"   riempimento verificato: immagine {D.VALORE_FILL_IMG} (bianco), "
          f"maschera {D.VALORE_FILL_MSK} ({C.CLASS_NAMES[D.VALORE_FILL_MSK]})")
    print(f"   pipeline verificata: etichette {sorted(visti)}, nessuna spuria")


def verifica_maschere(dataset_dir=None, n=5):
    """
    Self-check sul formato: gli ID devono essere 0..NUM_CLASSES-1 e 255.
    Valori come 29/150/179/212/226 significano che si sta leggendo la
    LUMINANZA della palette invece degli indici di classe.
    """
    dataset_dir = dataset_dir or C.DATASET_DIR
    print("\n[VERIFICA] Formato delle maschere...")
    valori, trovate = set(), 0
    for wsi in sorted(os.listdir(dataset_dir)):
        d = os.path.join(dataset_dir, wsi, C.CARTELLA_MSK)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d))[:2]:
            im = Image.open(os.path.join(d, f))
            v = np.unique(np.array(im))
            valori.update(v.tolist())
            print(f"   {wsi}/{f}: mode='{im.mode}' valori={v.tolist()}")
            trovate += 1
            if trovate >= n:
                break
        if trovate >= n:
            break

    attesi = set(range(C.N_CLASSI_FINI)) | {C.IGNORE_INDEX}
    if valori - attesi:
        raise RuntimeError(
            f"Valori inattesi nelle maschere: {sorted(valori - attesi)}. "
            f"Numeri tipo 29/150/179/212/226 indicano che si sta leggendo la "
            f"palette del PNG invece degli ID di classe.")
    print("   OK: valori compatibili con le classi attese.")


def verifica_cartelle():
    """Controlla che i percorsi esistano prima di iniziare."""
    if not os.path.isdir(C.DATASET_DIR):
        raise RuntimeError(f"DATASET_DIR non trovata: {C.DATASET_DIR}")
    wsi = [f for f in os.listdir(C.DATASET_DIR)
           if os.path.isdir(os.path.join(C.DATASET_DIR, f))
           and not f.startswith("_")]
    if not wsi:
        raise RuntimeError(f"Nessuna cartella WSI in {C.DATASET_DIR}")
    con_img = sum(1 for w in wsi
                  if os.path.isdir(os.path.join(C.DATASET_DIR, w, C.CARTELLA_IMG)))
    print(f"   WSI trovate: {len(wsi)}   con cartella "
          f"'{C.CARTELLA_IMG}': {con_img}")
    if con_img == 0:
        raise RuntimeError(
            f"Nessuna WSI contiene la cartella '{C.CARTELLA_IMG}'. "
            f"Verificare CARTELLA_IMG in config.py.")
    if con_img < len(wsi):
        print(f"   ATTENZIONE: {len(wsi)-con_img} WSI senza tile normalizzati, "
              f"verranno saltate.")
    C.crea_cartelle()
    print(f"   risultati in: {C.RISULTATI_DIR}")


# ====================================================================
# 2. ANALISI STRUTTURALE DEL DATASET
# ====================================================================
def analizza_dataset(campioni, comp, salva=True):
    """
    Fissa l'aspettativa PRIMA di guardare le metriche.

    Su MTCHI la lezione e' stata netta: il numero di tile e' irrilevante,
    conta il numero di CAMPIONI INDIPENDENTI, e per classe. Con 4
    pazienti LSIL la recall variava fra 0% e 77% a seconda di quali 3
    finivano in training, e quella varianza non era rumore da mediare ma
    la misura stessa.
    """
    px_wsi = D.pixel_per_wsi(campioni, comp)
    presenza = D.presenza_classi(px_wsi)
    n_cls = (comp > 0).sum(1)

    print("\n" + "=" * 78)
    print("  ANALISI STRUTTURALE DEL DATASET")
    print("=" * 78)
    print(f"  Tile                : {len(campioni):,}")
    print(f"  WSI                 : {len(px_wsi)}   "
          f"<-- il vero n dell'esperimento")
    print(f"  Normalizzazione     : {C.LIVELLO_NORMALIZZAZIONE} "
          f"(cartella '{C.CARTELLA_IMG}')")
    print(f"  Modalita' classi    : {C.MODALITA_CLASSI}")
    for k, etichetta in ((1, "mono-classe"), (2, "bi-classe"), (3, "3+ classi")):
        if k < 3:
            n = int((n_cls == k).sum())
        else:
            n = int((n_cls >= 3).sum())
        print(f"  Tile {etichetta:<15}: {n:>7,} ({100*n/max(len(campioni),1):>5.1f}%)")

    # ---------------- copertura per classe ----------------
    freq = comp.sum(0) / max(comp.sum(), 1)
    n_wsi_classe = {c: sum(1 for s in presenza.values() if c in s)
                    for c in range(C.NUM_CLASSES)}
    non_validabili = []

    print(f"\n  {'classe':<12}{'% pixel':>10}{'n WSI':>8}   stato")
    print("  " + "-" * 66)
    for c in range(C.NUM_CLASSES):
        n = n_wsi_classe[c]
        if n < C.N_FOLD:
            stato = "COPERTURA INSUFFICIENTE per i fold"
            non_validabili.append(c)
        elif n < C.SOGLIA_WSI_MINIME:
            stato = "copertura bassa: non validabile"
            non_validabili.append(c)
        else:
            stato = ""
        print(f"  {C.CLASS_NAMES[c]:<12}{100*freq[c]:>9.1f}%{n:>8}   {stato}")

    # ---------------- concentrazione ----------------
    print("\n  Concentrazione (WSI che contengono meta' dei pixel della classe):")
    concentrazione = {}
    for c in range(C.NUM_CLASSES):
        if c == C.CLASSE_SFONDO:
            continue
        v = sorted(((px_wsi[w][c], w) for w in px_wsi), reverse=True)
        tot = sum(x[0] for x in v)
        if tot == 0:
            continue
        acc, k, dominanti = 0.0, 0, []
        for val, w in v:
            acc += val
            k += 1
            dominanti.append(w)
            if acc >= 0.5 * tot:
                break
        concentrazione[C.CLASS_NAMES[c]] = {"n_wsi_meta_pixel": k,
                                            "wsi": dominanti}
        avviso = "  <-- estremamente concentrata" if k <= 2 else ""
        print(f"    {C.CLASS_NAMES[c]:<12}{k:>3} WSI  "
              f"({', '.join(dominanti[:3])}){avviso}")

    # ---------------- co-occorrenza ----------------
    print("\n  Co-occorrenza a livello di WSI:")
    idx = [c for c in range(C.NUM_CLASSES) if c != C.CLASSE_SFONDO]
    print("     " + "".join(f"{C.CLASS_NAMES[j][:9]:>11}" for j in idx))
    cooc = {}
    for i in idx:
        riga = ""
        for j in idx:
            n = sum(1 for s in presenza.values() if i in s and j in s)
            cooc[f"{C.CLASS_NAMES[i]}|{C.CLASS_NAMES[j]}"] = n
            riga += f"{n:>11}"
        print(f"  {C.CLASS_NAMES[i]:<10}{riga}")

    # ---------------- avvisi ----------------
    avvisi = []
    if C.MODALITA_CLASSI == 6:
        n_lsil_hsil = sum(1 for s in presenza.values() if 1 in s and 3 in s)
    else:
        n_lsil_hsil = sum(1 for s in presenza.values()
                          if C.IDX_LSIL_LAST in s and C.IDX_HSIL_LAST in s)

    print()
    if n_lsil_hsil < C.SOGLIA_COOCCORRENZA:
        msg = (f"CIN1 e HSIL coesistono in {n_lsil_hsil} WSI su "
               f"{len(px_wsi)}: il confine LSIL/HSIL NON e' validabile.")
        avvisi.append(msg)
        print(f"  ATTENZIONE: {msg}")
        print("  Le due classi sono quasi perfettamente separate per vetrino,")
        print("  quindi il modello puo' distinguerle riconoscendo il VETRINO")
        print("  anziche' la morfologia (Howard et al. 2021, Nat Commun 12:4423).")
        print("  Riportare come risultato esplorativo, mai come stima di")
        print("  popolazione.")

    for c in non_validabili:
        if c == C.CLASSE_SFONDO:
            continue
        msg = (f"{C.CLASS_NAMES[c]} presente in sole {n_wsi_classe[c]} WSI: "
               f"esclusa dalla metrica di selezione.")
        avvisi.append(msg)
        print(f"\n  ATTENZIONE: {msg}")
        print("  Resta addestrata e riportata, ma la sua media fra fold non")
        print("  descrive una popolazione. Riportare i valori per WSI.")

    print("=" * 78)

    diagnosi = {
        "n_tile": len(campioni),
        "n_wsi": len(px_wsi),
        "normalizzazione": C.LIVELLO_NORMALIZZAZIONE,
        "modalita_classi": C.MODALITA_CLASSI,
        "tile_mono_classe": int((n_cls == 1).sum()),
        "tile_bi_classe": int((n_cls == 2).sum()),
        "frequenza_pixel": {C.CLASS_NAMES[c]: float(freq[c])
                            for c in range(C.NUM_CLASSES)},
        "n_wsi_per_classe": {C.CLASS_NAMES[c]: n_wsi_classe[c]
                             for c in range(C.NUM_CLASSES)},
        "classi_non_validabili": [C.CLASS_NAMES[c] for c in non_validabili],
        "concentrazione": concentrazione,
        "cooccorrenza": cooc,
        "n_wsi_lsil_e_hsil": n_lsil_hsil,
        "avvisi": avvisi,
        "px_per_wsi": {w: v.tolist() for w, v in px_wsi.items()},
    }
    if salva:
        with open(C.path_metrica(C.NOME_DIAGNOSI), "w", encoding="utf-8") as f:
            json.dump(diagnosi, f, indent=2, ensure_ascii=False)
        print(f"\n  Diagnosi salvata in: {C.path_metrica(C.NOME_DIAGNOSI)}")

    return px_wsi, presenza, non_validabili, diagnosi