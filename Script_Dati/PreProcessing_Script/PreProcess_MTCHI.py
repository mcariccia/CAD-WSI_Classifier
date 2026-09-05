"""
PreProcess_MTCHI.py
====================================================================
Lancia PreProcess_Dataset.py sui tile di MTCHI, ereditando il target di
normalizzazione dal dataset privato.

PERCHE' UN WRAPPER E NON UNA COPIA MODIFICATA
----------------------------------------------
Il metodo deve restare identico riga per riga fra i due dataset,
altrimenti il confronto misura la procedura invece del dominio. Qui non
si duplica nulla: si importa lo stesso modulo, si cambiano due variabili
e si chiama la stessa funzione. NMF, restart, ALPHA_SPARSITA,
MAX_ITER_NMF, controllo del coseno, fallback su PRIOR_HE, I0 per
cartella: tutto invariato.

COSA CAMBIA, E PERCHE' NON INVALIDA IL CONFRONTO
-------------------------------------------------
Vahadane porta un'immagine DA una sorgente A una destinazione.

  sorgente     HE_src e I0, stimati per cartella. Restano dinamici,
               esattamente come nel privato: e' giusto che si adattino
               al vetrino che stanno leggendo.

  destinazione HE_target. Nel privato e' la mediana delle 40 WSI, ed e'
               l'aspetto su cui il modello ha imparato a leggere.

Ricalcolare la mediana su MTCHI cambierebbe la destinazione: le due
popolazioni finirebbero in punti diversi dello spazio colore e il
modello vedrebbe immagini che non assomigliano a quelle di
addestramento. Congelare il target e' cio' che mantiene il confronto
valido, non cio' che lo rompe.

C'e' anche una ragione pratica: matrici_valide raccoglie solo le stime
riuscite. Su MTCHI sono 6 su 20, quindi la mediana ricalcolata
definirebbe il target del dominio esterno usando il 30% dei suoi
vetrini.

PREREQUISITO — due righe in PreProcess_Dataset.py
--------------------------------------------------
1) fra le costanti in cima:

       HE_TARGET_ESTERNO = None    # None = comportamento attuale

2) in PASSATA B, il blocco esistente va dentro un else:

       if HE_TARGET_ESTERNO is not None:
           HE_target = HE_TARGET_ESTERNO
           print("\\nTarget ereditato dal dataset di addestramento:")
       else:
           if not matrici_valide:
               raise RuntimeError("Nessuna matrice valida trovata per calcolare la mediana!")
           HE_target = np.median(np.stack(matrici_valide), axis=0)
           HE_target /= np.linalg.norm(HE_target, axis=0)
           print("\\nMatrice Target Calcolata (Mediana del Dataset):")
       print(np.round(HE_target, 4))
       print("-" * 50)

Con HE_TARGET_ESTERNO = None il file si comporta bit per bit come
adesso: rilanciandolo sul privato si riottengono gli stessi tile. Il
ramo nuovo si attiva solo da qui.

USO
    python PreProcess_MTCHI.py
====================================================================
"""

import json
import os
import sys

import numpy as np

import PreProcess_Dataset as P

# ====================================================================
# PERCORSI  —  da verificare
# ====================================================================
DIR_TILE_MTCHI = r"E:\Tirocinio\Dataset\Dataset_Tiles_MTCHI"

# ATTENZIONE: deve essere il report del dataset con cui sono stati
# addestrati i modelli, cioe' quello a 512 (Dataset_Tiles), NON quello a
# 768 (Dataset_Tiles_768). Il target e' l'aspetto su cui il modello ha
# imparato: prenderlo da un run diverso porterebbe MTCHI verso una
# destinazione che il modello non conosce.
CHIAVE_TARGET = "HE_Target_Mediana"

HE_TARGET_PRIVATO = np.array([
    [0.62987723, 0.25336408],
    [0.69402265, 0.64861429],
    [0.34869362, 0.71770896],
], dtype=np.float64)


def riga(c="-", n=70):
    print(c * n)


# ====================================================================
# VERIFICHE
# ====================================================================
def carica_target(percorso):
    """
    Il target e' l'unico parametro che passa da un dataset all'altro:
    vale la pena controllarlo invece di fidarsi del file.
    """
    if not os.path.exists(percorso):
        raise SystemExit(f"Report privato non trovato:\n  {percorso}")

    with open(percorso, encoding="utf-8") as f:
        rep = json.load(f)

    if CHIAVE_TARGET not in rep:
        raise SystemExit(
            f"Chiave '{CHIAVE_TARGET}' assente in {percorso}.\n"
            f"Chiavi presenti: {list(rep.keys())}")

    M = np.array(rep[CHIAVE_TARGET], dtype=np.float64)
    if M.shape != (3, 2):
        raise SystemExit(f"Target di forma {M.shape}, attesa (3, 2)")

    # Le colonne devono essere versori: la trasformazione di Vahadane
    # assume vettori di stain a norma unitaria, e una colonna non
    # normalizzata cambierebbe silenziosamente l'intensita' del risultato.
    norme = np.linalg.norm(M, axis=0)
    if not np.allclose(norme, 1.0, atol=1e-6):
        print(f"  ATTENZIONE: colonne non unitarie {np.round(norme, 6)}, "
              f"rinormalizzo")
        M = M / norme

    # Contesto utile da riportare in tesi: quanto il target del privato si
    # discosta dal riferimento H&E di Ruifrok & Johnston (2001).
    prior = P.PRIOR_HE / np.linalg.norm(P.PRIOR_HE, axis=0)
    cos = np.sum(M * prior, axis=0)

    print("\n  Target ereditato dal dataset privato")
    print(f"  file      : {percorso}")
    print(f"  WSI valide: {rep.get('n_wsi_con_stima_valida', '?')} "
          f"su {len(rep.get('wsi', {})) or '?'}")
    print("  matrice   :")
    for r in np.round(M, 4):
        print(f"              [{r[0]:7.4f} {r[1]:7.4f}]")
    print(f"  coseno con il prior Ruifrok: H {cos[0]:.4f}   E {cos[1]:.4f}")
    return M


def verifica_dataset(percorso):
    if not os.path.isdir(percorso):
        raise SystemExit(f"Cartella dei tile non trovata:\n  {percorso}")

    cartelle = sorted(f for f in os.listdir(percorso)
                      if os.path.isdir(os.path.join(percorso, f))
                      and not f.startswith("_"))
    if not cartelle:
        raise SystemExit(
            f"Nessuna sottocartella in {percorso}.\n"
            f"PreProcess_Dataset stima UNA matrice per sottocartella: serve "
            f"una cartella per RoI, non una cartella piatta di tile.\n"
            f"Eseguire prima extraction_tiles.py.")

    senza, n_tile = [], 0
    for c in cartelle:
        d = os.path.join(percorso, c, P.CARTELLA_INPUT)
        if not os.path.isdir(d):
            senza.append(c)
            continue
        n_tile += sum(1 for f in os.listdir(d) if f.endswith(P.ESTENSIONE))

    print(f"\n  Dataset da normalizzare")
    print(f"  cartella  : {percorso}")
    print(f"  RoI       : {len(cartelle)}   tile: {n_tile}")
    if senza:
        print(f"  ATTENZIONE: {len(senza)} cartelle senza '{P.CARTELLA_INPUT}': "
              f"{senza[:3]}")
        print(f"  Verranno saltate da PreProcess_Dataset.")
    return cartelle


# ====================================================================
# MAIN
# ====================================================================
def main():
    riga("=")
    print("  PREPROCESSING VAHADANE SU MTCHI — target congelato")
    riga("=")

    if not hasattr(P, "HE_TARGET_ESTERNO"):
        raise SystemExit(
            "PreProcess_Dataset.py non espone HE_TARGET_ESTERNO.\n"
            "Manca la patch: aggiungere 'HE_TARGET_ESTERNO = None' fra le\n"
            "costanti e il ramo corrispondente in PASSATA B (vedi docstring).")

    target = HE_TARGET_PRIVATO
    print("\n  Target del dataset privato (36/40 WSI valide)")
    for r in np.round(target, 4):
        print(f"              [{r[0]:7.4f} {r[1]:7.4f}]")
    print()
    verifica_dataset(DIR_TILE_MTCHI)

    # I parametri del metodo NON si toccano: il confronto fra i due dataset
    # ha senso solo se la procedura e' identica. Si stampano per averli nel
    # log accanto ai risultati.
    print("\n  Parametri (invariati rispetto al privato)")
    for nome in ("ALPHA_SPARSITA", "MAX_ITER_NMF", "TOLLERANZA_ERRORE_NMF",
                 "MAX_PIXEL_FIT", "PIXEL_PER_TILE_CAMPIONE", "N_TILE_STIMA",
                 "SOGLIA_TESSUTO_GREZZA", "SOGLIA_ALLARME_COSENO", "SEED"):
        if hasattr(P, nome):
            print(f"    {nome:<26} {getattr(P, nome)}")

    # Passata C fa gia' shutil.rmtree su images_preprocessed prima di
    # riscriverla: non serve cancellare niente a mano.
    P.DATASET_DIR = DIR_TILE_MTCHI
    P.HE_TARGET_ESTERNO = target

    print()
    riga()
    P.process_dataset()
    riga()

    # ----------------------------------------------------------------
    # Il tasso di fallback e' un risultato, non un intoppo
    # ----------------------------------------------------------------
    p_rep = os.path.join(DIR_TILE_MTCHI, P.NOME_REPORT)
    if os.path.exists(p_rep):
        with open(p_rep, encoding="utf-8") as f:
            rep = json.load(f)
        n_tot = len(rep.get("wsi", {}))
        n_ok = rep.get("n_wsi_con_stima_valida", 0)
        print(f"\n  Stime valide: {n_ok} su {n_tot}"
              f"   fallback: {n_tot - n_ok}")
        if n_tot and (n_tot - n_ok) / n_tot > 0.25:
            print("\n  Molte RoI ricadono sul prior H&E. Non e' un guasto: e' il")
            print("  controllo del coseno che segnala una stima non affidabile.")
            print("  Le RoI di MTCHI sono ritagli stretti sull'epitelio e lo")
            print("  stroma, che e' il compartimento eosinofilo, resta in gran")
            print("  parte fuori dal crop: senza pixel a prevalenza di eosina le")
            print("  due componenti diventano quasi collineari e la NMF non ha")
            print("  nulla da separare.")
            print("  Il tasso di fallback va confrontato con quello del privato")
            print("  e riportato: e' una misura diretta della distanza fra i due")
            print("  domini. Se la cella 'vahadane_si' non guadagna nulla su")
            print("  'vahadane_no', la spiegazione e' questa.")
    print()


if __name__ == "__main__":
    sys.exit(main())