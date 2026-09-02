"""
diagnostica.py
====================================================================
Verifiche da eseguire PRIMA del training, e analisi strutturale del
dataset.

Filosofia: fallire subito e rumorosamente. Gli errori che hanno fatto
perdere piu' tempo in questo progetto erano tutti SILENZIOSI —
parametri di augmentation scartati con un warning, maschere lette come
palette invece che come indici, patch applicata a una cartella diversa
da quella letta dal training, una classe ridefinita localmente che
oscurava l'import. Ognuno di essi lasciava girare il training fino in
fondo producendo numeri plausibili e sbagliati.

L'analisi del dataset serve a fissare l'ASPETTATIVA prima di guardare
le metriche: con classi presenti in poche WSI, un numero alto non
significa quello che sembra.
====================================================================
"""

import os
import json
import random
import warnings

import cv2
import numpy as np
from PIL import Image
import albumentations as A

import config as C
import dati as D
from hed_augment import HEDIntensita


# ====================================================================
# 1. VERIFICHE DI CORRETTEZZA
# ====================================================================
def verifica_moduli():
    """
    L'augmentation di colorazione deve arrivare da hed_augment.py. Una
    definizione locale in dati.py oscurerebbe l'import: le modifiche a
    hed_augment.py non avrebbero effetto e il training girerebbe lo
    stesso. E' gia' successo due volte.
    """
    if D.HEDIntensita.__module__ != "hed_augment":
        raise RuntimeError(
            f"dati.HEDIntensita risolve a '{D.HEDIntensita.__module__}' invece "
            f"che a 'hed_augment': esiste una definizione locale che oscura "
            f"l'import.")
    # I parametri di config devono combaciare con la firma della classe.
    try:
        HEDIntensita(**C.parametri_hed())
    except TypeError as e:
        raise RuntimeError(
            f"config.parametri_hed() non e' compatibile con HEDIntensita: {e}")
    print(f"   augmentation di colorazione: hed_augment.HEDIntensita, "
          f"K {C.K_INTENSITA}/{C.K_ATTENUAZIONE}/{C.K_RAPPORTO}")


def verifica_riempimento():
    """
    Tre controlli distinti.

    (1) DETERMINISTICO: rotate=45 e scale=1.0 garantiscono angoli vuoti,
        quindi se il riempimento e' attivo DEVE comparire il valore di
        fill, e i pixel bianchi dell'immagine devono corrispondere a
        etichette "vetro" nella maschera (coerenza semantica).
        Non si puo' usare affine() per questo: ha p=0.5 e scale fino a
        1.15, e quando la scala supera 1.0 l'immagine viene ingrandita e
        non si formano affatto zone vuote.

    (2) STATISTICO: molte passate della pipeline reale su una maschera
        che contiene ANCHE 254 e 255, verificando che non nascano
        etichette fuori dall'insieme atteso. Catturerebbe
        un'interpolazione non-nearest, che sui due valori alti
        produrrebbe valori intermedi inesistenti.

    (3) In piu' si intercettano i warning "not valid for transform": da
        albumentations 2.x i parametri sbagliati non sollevano eccezioni.
    """
    kw = D.kwargs_riempimento()

    img = np.full((128, 128, 3), 60, dtype=np.uint8)
    msk = np.full((128, 128), 3, dtype=np.uint8)          # HSIL
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

    bianchi = (out["image"] == D.VALORE_FILL_IMG).all(axis=2)
    if bianchi.any():
        etichette_bianchi = set(np.unique(out["mask"][bianchi]).tolist())
        if etichette_bianchi != {D.VALORE_FILL_MSK}:
            raise RuntimeError(
                f"Incoerenza fra riempimento immagine e maschera: i pixel "
                f"bianchi hanno etichette {sorted(etichette_bianchi)}, atteso "
                f"solo {D.VALORE_FILL_MSK} ({C.CLASS_NAMES[D.VALORE_FILL_MSK]}).")

    ammesse = list(range(C.NUM_CLASSES)) + [C.IGNORE_VETRO, C.IGNORE_INDEX]
    img2 = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    msk2 = np.random.choice(ammesse, size=(256, 256)).astype(np.uint8)
    tf = A.Compose(D.lista_augmentation())
    visti = set()
    with warnings.catch_warnings(record=True) as catturati:
        warnings.simplefilter("always")
        for _ in range(60):
            visti.update(np.unique(tf(image=img2, mask=msk2)["mask"]).tolist())

    spurie = visti - set(ammesse)
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
    Self-check sul formato dei file SU DISCO: gli ID devono essere
    0..N_CLASSI_FINI-1 e 255. Il 254 non compare mai qui, nasce a
    runtime in dati.marca_vetro.
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


def verifica_soglia_vetro(campioni, par_vetro=None, n=40, seed=0):
    """
    La pseudo-etichetta vetro/tessuto e' SUPERVISIONE, non un prior:
    va validata contro le annotazioni vere prima di usarla. Se la
    soglia e' tarata male, il training impara un confine falso e gira
    fino in fondo senza dare alcun segnale.

    (1) Test deterministico su un'immagine sintetica meta' bianca e
        meta' scura: marca_vetro deve produrre 254 sulla prima meta' e
        lasciare 255 sulla seconda.
    (2) Test statistico su tile reali: sui pixel annotati Sfondo la
        soglia deve dire "vetro", sui pixel annotati tessuto no.
    """
    par = C.parametri_vetro() if par_vetro is None else par_vetro
    soglia = par["soglia_od"]

    finta = np.zeros((16, 32, 3), dtype=np.uint8)
    finta[:, :16] = 250                                   # vetro
    finta[:, 16:] = 120                                   # tessuto
    m = np.full((16, 32), par["ignore_index"], dtype=np.uint8)
    m = D.marca_vetro(finta, m, **par)
    if not (np.all(m[:, :16] == par["valore_vetro"])
            and np.all(m[:, 16:] == par["ignore_index"])):
        raise RuntimeError(
            "marca_vetro non separa un'immagine sintetica bianco/scuro: "
            "controllare SOGLIA_OD_VETRO e il calcolo della densita' ottica.")

    scelti = random.Random(seed).sample(campioni, min(n, len(campioni)))
    vetro_su_sfondo = tot_sfondo = 0
    vetro_su_tessuto = tot_tessuto = 0
    quota_ignoti = []

    for p_img, p_msk, _ in scelti:
        bgr = cv2.imread(p_img)
        if bgr is None:
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        msk = D.leggi_maschera(p_msk)
        od = -np.log(np.clip(img.astype(np.float32) / 255.0, 1e-3, 1.0)).mean(2)
        e_vetro = od < soglia

        s = msk == C.CLASSE_SFONDO
        t = (msk > C.CLASSE_SFONDO) & (msk < C.NUM_CLASSES)
        tot_sfondo += int(s.sum())
        vetro_su_sfondo += int((e_vetro & s).sum())
        tot_tessuto += int(t.sum())
        vetro_su_tessuto += int((e_vetro & t).sum())
        ig = msk == C.IGNORE_INDEX
        if ig.any():
            quota_ignoti.append(float((e_vetro & ig).sum()) / float(ig.sum()))

    sens = vetro_su_sfondo / max(tot_sfondo, 1)
    err = vetro_su_tessuto / max(tot_tessuto, 1)
    print(f"\n[VERIFICA] Pseudo-etichetta vetro (soglia OD {soglia}, "
          f"{len(scelti)} tile)")
    print(f"   Sfondo annotato riconosciuto vetro     : {100*sens:.1f}%")
    print(f"   tessuto annotato scambiato per vetro   : {100*err:.1f}%")
    if quota_ignoti:
        print(f"   quota di vetro nelle zone non annotate : "
              f"{100*np.mean(quota_ignoti):.1f}%")
    if sens < 0.85 or err > 0.10:
        raise RuntimeError(
            f"SOGLIA_OD_VETRO={soglia} non separa vetro e tessuto "
            f"(sensibilita' {sens:.2f}, errore {err:.2f}). Alzarla se troppo "
            f"tessuto viene scambiato per vetro, abbassarla nel caso opposto.")
    print("   OK: la soglia separa vetro e tessuto.")
    return {"sensibilita_sfondo": sens, "errore_tessuto": err,
            "quota_vetro_non_annotato": (float(np.mean(quota_ignoti))
                                         if quota_ignoti else None)}


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
    if os.path.isdir(C.RISULTATI_DIR) and os.listdir(C.RISULTATI_DIR):
        print(f"   ATTENZIONE: {C.RISULTATI_DIR} non e' vuota: questo run "
              f"sovrascrivera' i file omonimi.")
    C.crea_cartelle()
    print(f"   run '{C.NOME_RUN}' -> {C.RISULTATI_DIR}")


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
        n = int((n_cls == k).sum()) if k < 3 else int((n_cls >= 3).sum())
        print(f"  Tile {etichetta:<15}: {n:>7,} "
              f"({100*n/max(len(campioni),1):>5.1f}%)")

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
               f"esclusa dalla mIoU riportata a schermo.")
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