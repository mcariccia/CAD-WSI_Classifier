"""
splitter_vincolato.py
====================================================================
Costruzione della partizione train/test a livello di WSI.

DIFFERENZE RISPETTO ALLA VERSIONE PRECEDENTE
--------------------------------------------
1. L'obiettivo NON include Sfondo. La classe 0 non e' un'annotazione:
   e' il background del LabeledImageServer, cioe' tutto cio' che il
   patologo non ha toccato. Bilanciarla consuma un sesto della funzione
   obiettivo per una quantita' priva di significato diagnostico.

2. L'obiettivo NON include Stroma. Con 636M di pixel di stroma di cui
   il 43% in Prova 4 e il 40% in Prova 2, le quote raggiungibili con 8
   vetrini sono <=17% (nessun gigante) oppure >=40% (un gigante): il
   target del 20% cade in un buco. Nella versione precedente il 99,8%
   dell'errore totale all'ottimo era il solo termine Stroma, cioe'
   l'intera ottimizzazione era decisa da un termine irriducibile che
   trascinava nel test il vetrino piu' grande del dataset. Qui lo
   Stroma e' VINCOLATO (tetto), non targettato.

3. Vincoli di PRESENZA, non solo di quota di pixel. Una classe puo'
   raggiungere il 20% venendo tutta da un vetrino solo: in quel caso la
   variabilita' fra vetrini, l'unica che conta con n dell'ordine delle
   decine, non e' stimabile. La presenza usa la stessa soglia di
   dati.presenza_classi (SOGLIA_PRESENZA_CLASSE), non ">0 pixel".

4. Vincolo di VALIDABILITA' IN CV. diagnostica dichiara non validabile
   una classe presente in meno di 2*N_FOLD WSI. Spostare vetrini nel
   test riduce quel conteggio nel pool di CV: con CIN1 in 12 WSI, il
   test ne puo' prendere al massimo 2 prima che CIN1 diventi non
   validabile in cross-validation. E' il vincolo che lega direttamente
   la scelta dello split alla metrica primaria.

5. Vincolo di DOMINANZA. Nessun vetrino di test puo' superare una quota
   fissata dei pixel di tessuto del test, altrimenti ogni metrica
   aggregata per pixel misura quel vetrino e non il modello.

6. NIENTE ARGMIN ESATTO. Con decine di milioni di candidati e un
   obiettivo liscio esistono migliaia di partizioni entro epsilon
   dall'ottimo: prendere l'argmin significa lasciare che la quinta cifra
   decimale scelga i vetrini. Si tiene una rosa dei migliori e si
   ordina per criteri secondari dichiarati.

7. PRESPECIFICAZIONE. Lo script serializza in JSON l'intero insieme dei
   vincoli, la funzione obiettivo, il numero di partizioni ammissibili e
   un hash della configurazione. E' quello che serve per scrivere in
   tesi che la regola di partizione e' stata dichiarata prima della
   selezione, invece di descriverla a posteriori.

LIMITE NON RISOLVIBILE QUI
---------------------------
La protezione dal leakage a livello di PAZIENTE non e' implementabile
finche' l'identita' del paziente non e' recuperabile dai nomi "Prova N".
Il gancio c'e' (MAPPA_PAZIENTI): se la mappa arriva, i vetrini dello
stesso paziente vengono trattati come un blocco indivisibile. Finche'
resta vuota, va dichiarato come limite.
====================================================================
"""

import hashlib
import itertools
import json
import time
from math import comb

import numpy as np
import pandas as pd

# ====================================================================
# CONFIGURAZIONE
# ====================================================================
FILE_PROFILO = "profilo_wsi.json"
FILE_TRAIN = "train_split_vincolato.csv"
FILE_TEST = "test_split_vincolato.csv"
FILE_REGOLA = "regola_split.json"

NOMI_CLASSI = ["Sfondo", "CIN1", "Ghiandole", "HSIL", "Mucosa", "Stroma"]
IDX_SFONDO = 0
IDX_CIN1, IDX_GHIANDOLE, IDX_HSIL, IDX_MUCOSA, IDX_STROMA = 1, 2, 3, 4, 5

# Classi su cui si bilancia la quota di pixel. Sfondo escluso (vetro),
# Stroma escluso (target irraggiungibile, vedi docstring).
CLASSI_OBIETTIVO = [IDX_CIN1, IDX_GHIANDOLE, IDX_HSIL, IDX_MUCOSA]

N_TEST = 8
QUOTA_TARGET = N_TEST / 40.0          # coerente con la dimensione del test

# Deve coincidere con dati.presenza_classi / config.SOGLIA_PRESENZA_CLASSE.
SOGLIA_PRESENZA = 0.005

# Deve coincidere con config.SOGLIA_WSI_MINIME = 2 * N_FOLD.
N_FOLD = 5
WSI_MINIME_TRAIN = 2 * N_FOLD

# Deroga esplicita, per classe, al pavimento 2*N_FOLD. Vuota per scelta:
# con i dati attuali non serve. Se in futuro un vincolo risultasse
# infattibile, la deroga va messa QUI e non aggirata abbassando la presenza
# minima nel test, cosi' resta visibile e finisce serializzata in
# regola_split.json.
DEROGHE_WSI_MINIME = {}


def wsi_minime(classe):
    return DEROGHE_WSI_MINIME.get(classe, WSI_MINIME_TRAIN)

# --- vincoli clinici ---------------------------------------------------
# Vetrini forzati nel train per motivi di qualita' del dato: annotazioni
# rattoppate a mano, artefatti di colorazione noti. Il test set deve avere
# il ground truth piu' pulito disponibile.
WSI_FORZATE_TRAIN = []                # es. ["Prova 20", "Prova 59"]

# Classi che devono restare validabili in cross-validation: il pool di
# training deve conservarne almeno WSI_MINIME_TRAIN vetrini.
CLASSI_DA_VALIDARE = [IDX_CIN1, IDX_GHIANDOLE, IDX_HSIL, IDX_MUCOSA, IDX_STROMA]

# Presenza minima nel test, per classe. Sotto 2 vetrini nessun intervallo
# di confidenza per WSI e' calcolabile.
PRESENZA_MINIMA_TEST = {IDX_CIN1: 2, IDX_GHIANDOLE: 3, IDX_HSIL: 3,
                        IDX_MUCOSA: 3, IDX_STROMA: 1}

# Almeno un vetrino privo di qualunque lesione: senza, la specificita' si
# stima solo sul tessuto non lesionale DENTRO vetrini lesionali, che e' una
# popolazione diversa da quella di un caso negativo.
MIN_WSI_NEGATIVE_TEST = 1

# Almeno un vetrino con interfaccia lesione / mucosa normale: e' il confine
# su cui il patologo definisce l'estensione della lesione.
MIN_WSI_INTERFACCIA_TEST = 3

# Potenza statistica: pixel assoluti minimi per classe nel test.
PIXEL_MINIMI_TEST = {IDX_CIN1: 5_000_000, IDX_HSIL: 10_000_000,
                     IDX_MUCOSA: 5_000_000, IDX_GHIANDOLE: 2_000_000}

# --- vincoli statistici ------------------------------------------------
MAX_DOMINANZA_TEST = 0.35     # quota max dei pixel di tessuto di test in un WSI
MAX_QUOTA_STROMA = 0.25       # tetto, non target

# --- selezione ---------------------------------------------------------
DIM_ROSA = 500                # candidati tenuti prima dei criteri secondari
TOLLERANZA_ROSA = 1.15        # entro il 15% dell'MSE minimo

# Gancio per il futuro: {"Prova 3": "PZ001", ...}. Vuoto = nessuna
# protezione a livello di paziente, da dichiarare come limite.
MAPPA_PAZIENTI = {}

BLOCCO = 500_000              # dimensione del blocco di enumerazione


# ====================================================================
class OttimizzatoreSplit:

    def __init__(self, file_profilo=FILE_PROFILO):
        with open(file_profilo, "r", encoding="utf-8") as f:
            profili = json.load(f)

        # sorted() e non keys(): l'ordine di un JSON non e' garantito e
        # uno split "ottimo" che dipende dall'ordine di lettura non e'
        # riproducibile.
        self.wsi = sorted(profili, key=lambda s: (len(s), s))
        self.n = len(self.wsi)
        self.M = np.array([[profili[w].get(str(c), 0) for c in range(6)]
                           for w in self.wsi], dtype=np.float64)

        self.totali = np.where(self.M.sum(0) == 0, 1.0, self.M.sum(0))
        self.tessuto = self.M[:, 1:].sum(1)

        quote_interne = self.M / np.maximum(self.M.sum(1, keepdims=True), 1.0)
        self.presenza = quote_interne >= SOGLIA_PRESENZA

        self.negativa = ~self.presenza[:, IDX_CIN1] & ~self.presenza[:, IDX_HSIL]
        self.interfaccia = (self.presenza[:, IDX_MUCOSA] &
                            (self.presenza[:, IDX_CIN1] |
                             self.presenza[:, IDX_HSIL]))
        self.cooccorrenza = self.presenza[:, IDX_CIN1] & self.presenza[:, IDX_HSIL]

        self._blocca()

    # ----------------------------------------------------------------
    def _blocca(self):
        """
        Vetrini esclusi dal pool candidabile al test.

        La co-occorrenza CIN1+HSIL e' RILEVATA, non scritta a mano: se il
        dataset cambia, il vincolo si aggiorna da solo. Un vetrino con
        transizione di grado nel test lascia il training senza nessuna
        tile che contenga entrambi i gradi, cioe' esattamente il
        meccanismo che su MTCHI produceva IoU LSIL prossima a zero nelle
        RoI di co-occorrenza.
        """
        bloccati = set(WSI_FORZATE_TRAIN)
        bloccati |= {self.wsi[i] for i in np.where(self.cooccorrenza)[0]}

        if MAPPA_PAZIENTI:
            pazienti_bloccati = {MAPPA_PAZIENTI[w] for w in bloccati
                                 if w in MAPPA_PAZIENTI}
            bloccati |= {w for w, pz in MAPPA_PAZIENTI.items()
                         if pz in pazienti_bloccati}

        self.bloccati = sorted(bloccati, key=lambda s: (len(s), s))
        self.candidati = [i for i, w in enumerate(self.wsi)
                          if w not in bloccati]

    # ----------------------------------------------------------------
    def diagnosi(self):
        """
        Fattibilita' PRIMA di enumerare. Un vincolo infattibile va visto
        qui, non dopo due minuti di ricerca conclusa senza soluzioni.
        """
        print("=" * 72)
        print("  DIAGNOSI DI FATTIBILITA'")
        print("=" * 72)
        print(f"  WSI totali: {self.n}   test richiesto: {N_TEST}")
        print(f"  bloccate nel train: {', '.join(self.bloccati) or 'nessuna'}")
        if not MAPPA_PAZIENTI:
            print("  ATTENZIONE: mappa vetrino->paziente assente. Nessuna")
            print("  protezione dal leakage a livello di paziente. Da dichiarare.")

        print(f"\n  {'classe':<12}{'WSI tot':>9}{'min train':>11}"
              f"{'max nel test':>14}{'min nel test':>14}{'esito':>10}")
        fattibile = True
        for c in CLASSI_DA_VALIDARE:
            tot = int(self.presenza[:, c].sum())
            bloccate_con_c = sum(1 for w in self.bloccati
                                 if self.presenza[self.wsi.index(w), c])
            # Il test puo' prendere al massimo (tot - pavimento sul train)
            # vetrini della classe, e comunque non piu' di quelli non
            # bloccati: i bloccati contano gia' dentro il pavimento.
            massimo = min(tot - wsi_minime(c), tot - bloccate_con_c, N_TEST)
            massimo = max(massimo, 0)
            minimo = PRESENZA_MINIMA_TEST.get(c, 0)
            ok = minimo <= massimo
            fattibile &= ok
            print(f"  {NOMI_CLASSI[c]:<12}{tot:>9}{wsi_minime(c):>11}"
                  f"{massimo:>14}{minimo:>14}"
                  f"{'ok' if ok else 'CONFLITTO':>10}")

        n_neg = int(self.negativa.sum())
        n_int = int(self.interfaccia.sum())
        print(f"\n  WSI puramente negative (ne CIN1 ne HSIL): {n_neg}"
              f"   richieste nel test: {MIN_WSI_NEGATIVE_TEST}")
        if n_neg <= 3:
            print("    Con cosi' poche, la specificita' si stima quasi tutta su")
            print("    tessuto non lesionale dentro vetrini lesionali. Limite da")
            print("    riportare esplicitamente.")
        print(f"  WSI con interfaccia lesione/mucosa: {n_int}"
              f"   richieste nel test: {MIN_WSI_INTERFACCIA_TEST}")

        quote = self.M[:, IDX_STROMA] / self.totali[IDX_STROMA]
        ordina = np.argsort(-quote)[:3]
        print("\n  Concentrazione dello Stroma (perche' non e' targettabile):")
        for i in ordina:
            print(f"    {self.wsi[i]:<12}{100*quote[i]:>6.1f}% dello stroma totale")
        print(f"    tetto imposto al test: {100*MAX_QUOTA_STROMA:.0f}%")

        print(f"\n  combinazioni da valutare: "
              f"{comb(len(self.candidati), N_TEST):,}")
        print("=" * 72)
        if not fattibile:
            raise RuntimeError(
                "Vincoli infattibili: la presenza minima richiesta nel test "
                "supera il massimo compatibile con la validabilita' in CV. "
                "Abbassare PRESENZA_MINIMA_TEST oppure N_FOLD.")
        return fattibile

    # ----------------------------------------------------------------
    def _valuta_blocco(self, A):
        """
        Obiettivo e vincoli su un blocco di combinazioni.

        Si somma iterando sulle N_TEST posizioni invece di indicizzare
        (B, N_TEST, 6) in un colpo solo: l'array intermedio sarebbe
        centinaia di MB per blocco senza alcun guadagno di velocita'.
        """
        B = A.shape[0]
        quote = np.zeros((B, 6), dtype=np.float64)
        pixel = np.zeros((B, 6), dtype=np.float64)
        pres = np.zeros((B, 6), dtype=np.int16)
        neg = np.zeros(B, dtype=np.int16)
        interf = np.zeros(B, dtype=np.int16)
        tess = np.zeros((B, N_TEST), dtype=np.float64)

        for j in range(N_TEST):
            idx = self.c_idx[A[:, j]]
            quote += self.V[idx]
            pixel += self.MPX[idx]
            pres += self.PRES[idx]
            neg += self.NEG[idx]
            interf += self.INT[idx]
            tess[:, j] = self.TESS[idx]

        errore = ((quote[:, CLASSI_OBIETTIVO] - QUOTA_TARGET) ** 2).sum(1)

        vietato = np.zeros(B, dtype=bool)
        vietato |= tess.max(1) / np.maximum(tess.sum(1), 1.0) > MAX_DOMINANZA_TEST
        vietato |= quote[:, IDX_STROMA] > MAX_QUOTA_STROMA
        vietato |= neg < MIN_WSI_NEGATIVE_TEST
        vietato |= interf < MIN_WSI_INTERFACCIA_TEST
        for c, m in PRESENZA_MINIMA_TEST.items():
            vietato |= pres[:, c] < m
        for c, m in PIXEL_MINIMI_TEST.items():
            vietato |= pixel[:, c] < m
        for c in CLASSI_DA_VALIDARE:
            vietato |= (self.presenza[:, c].sum() - pres[:, c]) < wsi_minime(c)

        return errore, vietato, tess

    # ----------------------------------------------------------------
    def cerca(self):
        self.c_idx = np.array(self.candidati)
        self.V = self.M / self.totali
        self.MPX = self.M
        self.PRES = self.presenza.astype(np.int16)
        self.NEG = self.negativa.astype(np.int16)
        self.INT = self.interfaccia.astype(np.int16)
        self.TESS = self.tessuto

        n_cand = len(self.candidati)
        totale = comb(n_cand, N_TEST)
        print(f"\n  Enumerazione esatta di {totale:,} combinazioni...")

        rosa_err = np.full(DIM_ROSA, np.inf)
        rosa_idx = np.zeros((DIM_ROSA, N_TEST), dtype=np.int16)
        n_ammissibili = 0
        t0 = time.time()

        iteratore = itertools.combinations(range(n_cand), N_TEST)
        while True:
            blocco = list(itertools.islice(iteratore, BLOCCO))
            if not blocco:
                break
            A = np.array(blocco, dtype=np.int16)
            errore, vietato, _ = self._valuta_blocco(A)

            buoni = ~vietato
            n_ammissibili += int(buoni.sum())
            if not buoni.any():
                continue

            e, a = errore[buoni], A[buoni]
            k = min(DIM_ROSA, e.shape[0])
            sel = np.argpartition(e, k - 1)[:k]

            tutte_e = np.concatenate([rosa_err, e[sel]])
            tutte_a = np.concatenate([rosa_idx, a[sel]])
            ordine = np.argsort(tutte_e)[:DIM_ROSA]
            rosa_err, rosa_idx = tutte_e[ordine], tutte_a[ordine]

        print(f"  completata in {time.time() - t0:.0f} s")
        print(f"  partizioni ammissibili: {n_ammissibili:,} "
              f"({100 * n_ammissibili / totale:.3f}% dello spazio)")
        if n_ammissibili == 0:
            raise RuntimeError("Nessuna partizione soddisfa i vincoli.")

        self.rosa_err, self.rosa_idx = rosa_err, rosa_idx
        self.n_ammissibili = n_ammissibili
        return self._scegli()

    # ----------------------------------------------------------------
    def _scegli(self):
        """
        Fra le partizioni entro TOLLERANZA_ROSA dall'MSE minimo si sceglie
        con criteri secondari DICHIARATI, in ordine lessicografico:
          1. dominanza del vetrino piu' grande, minima
          2. copertura di classe nel test, massima
          3. MSE, minimo
        Riportare quante partizioni erano equivalenti e' parte del
        risultato: dice che la scelta non e' un ottimo isolato.
        """
        valide = np.isfinite(self.rosa_err)
        err = self.rosa_err[valide]
        idx = self.rosa_idx[valide]
        soglia = err.min() * TOLLERANZA_ROSA + 1e-12
        entro = err <= soglia
        err, idx = err[entro], idx[entro]

        punteggi = []
        for r in range(idx.shape[0]):
            sel = self.c_idx[idx[r]]
            t = self.tessuto[sel]
            dom = float(t.max() / max(t.sum(), 1.0))
            cop = int(self.presenza[sel][:, 1:].sum())
            punteggi.append((round(dom, 3), -cop, float(err[r]), r))
        punteggi.sort()
        migliore = punteggi[0]

        self.n_equivalenti = int(idx.shape[0])
        self.errore = migliore[2]
        self.idx_test = sorted(self.c_idx[idx[migliore[3]]].tolist())
        self.idx_train = [i for i in range(self.n) if i not in self.idx_test]
        return self.idx_train, self.idx_test

    # ----------------------------------------------------------------
    def riepilogo(self):
        tr, te = self.idx_train, self.idx_test
        q_te = self.M[te].sum(0) / self.totali
        q_tr = self.M[tr].sum(0) / self.totali
        t = self.tessuto[te]

        print("\n" + "=" * 72)
        print("  PARTIZIONE SELEZIONATA")
        print("=" * 72)
        print(f"  MSE sulle classi obiettivo: {self.errore:.3e}")
        print(f"  partizioni entro il {100*(TOLLERANZA_ROSA-1):.0f}% "
              f"dell'ottimo: {self.n_equivalenti}")
        print(f"  train {len(tr)} WSI   test {len(te)} WSI")
        print(f"\n  test: {', '.join(self.wsi[i] for i in te)}")

        print(f"\n  {'classe':<12}{'train %':>10}{'test %':>10}"
              f"{'px test':>16}{'WSI train':>11}{'WSI test':>10}")
        for c in range(6):
            marca = " *" if c in CLASSI_OBIETTIVO else "  "
            print(f"  {NOMI_CLASSI[c]:<12}{100*q_tr[c]:>9.2f}%"
                  f"{100*q_te[c]:>9.2f}%{int(self.M[te, c].sum()):>16,}"
                  f"{int(self.presenza[tr][:, c].sum()):>11}"
                  f"{int(self.presenza[te][:, c].sum()):>10}{marca}")
        print("  * = classe inclusa nella funzione obiettivo")

        i_dom = te[int(np.argmax(t))]
        print(f"\n  vetrino dominante nel test: {self.wsi[i_dom]} "
              f"({100*t.max()/t.sum():.1f}% del tessuto, tetto "
              f"{100*MAX_DOMINANZA_TEST:.0f}%)")
        print(f"  WSI puramente negative nel test: "
              f"{int(self.negativa[te].sum())}")
        print(f"  WSI con interfaccia lesione/mucosa nel test: "
              f"{int(self.interfaccia[te].sum())}")
        print(f"  WSI di co-occorrenza CIN1+HSIL nel test: "
              f"{int(self.cooccorrenza[te].sum())} (deve essere 0)")
        print("=" * 72)

    # ----------------------------------------------------------------
    def salva(self):
        train = [self.wsi[i] for i in self.idx_train]
        test = [self.wsi[i] for i in self.idx_test]
        pd.DataFrame(train, columns=["WSI"]).to_csv(FILE_TRAIN, index=False)
        pd.DataFrame(test, columns=["WSI"]).to_csv(FILE_TEST, index=False)

        regola = {
            "obiettivo": {
                "tipo": "MSE sulla quota di pixel per classe",
                "classi": [NOMI_CLASSI[c] for c in CLASSI_OBIETTIVO],
                "target": QUOTA_TARGET,
                "escluse": {
                    "Sfondo": "vetro, non e' un'annotazione diagnostica",
                    "Stroma": "target irraggiungibile, vincolato con un tetto",
                },
            },
            "vincoli": {
                "n_test": N_TEST,
                "soglia_presenza": SOGLIA_PRESENZA,
                "wsi_minime_train_default": WSI_MINIME_TRAIN,
                "deroghe_wsi_minime": {NOMI_CLASSI[c]: v for c, v
                                       in DEROGHE_WSI_MINIME.items()},
                "classi_da_validare": [NOMI_CLASSI[c] for c in CLASSI_DA_VALIDARE],
                "presenza_minima_test": {NOMI_CLASSI[c]: v for c, v
                                         in PRESENZA_MINIMA_TEST.items()},
                "pixel_minimi_test": {NOMI_CLASSI[c]: v for c, v
                                      in PIXEL_MINIMI_TEST.items()},
                "min_wsi_negative_test": MIN_WSI_NEGATIVE_TEST,
                "min_wsi_interfaccia_test": MIN_WSI_INTERFACCIA_TEST,
                "max_dominanza_test": MAX_DOMINANZA_TEST,
                "max_quota_stroma": MAX_QUOTA_STROMA,
                "wsi_forzate_train": WSI_FORZATE_TRAIN,
                "cooccorrenza_forzata_train": [
                    self.wsi[i] for i in np.where(self.cooccorrenza)[0]],
            },
            "selezione": {
                "dim_rosa": DIM_ROSA,
                "tolleranza": TOLLERANZA_ROSA,
                "criteri_secondari": ["dominanza minima",
                                      "copertura di classe massima",
                                      "MSE minimo"],
            },
            "risultato": {
                "train": train,
                "test": test,
                "mse": self.errore,
                "partizioni_ammissibili": self.n_ammissibili,
                "partizioni_equivalenti": self.n_equivalenti,
            },
            "limiti": {
                "leakage_paziente": bool(MAPPA_PAZIENTI),
                "nota": ("Senza mappa vetrino->paziente non e' possibile "
                         "escludere che vetrini dello stesso paziente cadano "
                         "su lati opposti della partizione."),
            },
        }
        impronta = hashlib.sha256(
            json.dumps({k: regola[k] for k in ("obiettivo", "vincoli",
                                               "selezione")},
                       sort_keys=True).encode()).hexdigest()[:16]
        regola["hash_configurazione"] = impronta

        with open(FILE_REGOLA, "w", encoding="utf-8") as f:
            json.dump(regola, f, indent=2, ensure_ascii=False)

        print(f"\n  {FILE_TRAIN}")
        print(f"  {FILE_TEST}")
        print(f"  {FILE_REGOLA}   hash {impronta}")


# ====================================================================
def main():
    opt = OttimizzatoreSplit()
    opt.diagnosi()
    opt.cerca()
    opt.riepilogo()
    opt.salva()


if __name__ == "__main__":
    main()