"""
test.py
====================================================================
Valutazione FINALE sulle WSI tenute fuori dalla cross-validation.

DA ESEGUIRE UNA VOLTA SOLA, alla fine.
---------------------------------------
Se si usa il test set per scegliere iperparametri, o si rilancia
cambiando qualcosa dopo aver visto i numeri, smette di essere un test e
diventa un secondo validation set. La stima che produce non e' piu' una
stima di generalizzazione.

Se serve provare varianti, si provano in cross-validation.

PERCHE' NON SI SCEGLIE "IL FOLD MIGLIORE"
------------------------------------------
I punteggi dei fold sono punteggi di VALIDATION. Prenderne il massimo e
riportarlo e' selezione sui dati di validazione. Su questo run:

    fold 2   0.7720   <-- il massimo
    fold 3   0.7421       (RIPIEGO)
    fold 0   0.7439
    fold 4   0.7101       (RIPIEGO)
    fold 1   0.6634
    media    0.7263   <-- la stima onesta

I cinque numeri non sono nemmeno confrontabili fra loro: la
composizione degli insiemi di validazione varia di un fattore sei in
volume e di un fattore trenta nella quota di LSIL. Il fold 2 non e' il
modello migliore, e' il fold piu' fortunato.

FOLD DI RIPIEGO
----------------
Un fold in cui nessuna epoca ha rispettato l'operating point dichiarato
viene selezionato su mIoU. Il suo modello NON e' calibrato allo stesso
punto di lavoro degli altri, e mescolarlo nell'ensemble senza dirlo
rende l'ensemble non descrivibile. Lo script legge dal report di CV
quali fold sono ammissibili e produce, oltre all'ensemble completo, un
ensemble dei soli fold ammissibili come analisi di sensibilita'.

DUE DECISIONI
--------------
  argmax    : minimizza l'errore 0-1. Tratta CIN1 -> Mucosa e
              CIN1 -> HSIL come lo stesso errore, che clinicamente non
              sono: il primo perde la lesione, il secondo la sovrastima.
  costo     : decisione a rischio atteso minimo sotto la matrice
              MATRICE_COSTI (Elkan 2001), la STESSA gia' usata nella
              loss.  y = argmin_j sum_i p_i * Omega[i, j]

La matrice e' PRESPECIFICATA: e' una dichiarazione di preferenze
cliniche scritta prima di vedere il test, non un parametro stimato sui
suoi dati. Usarla non consuma il test set. Ritoccarla DOPO aver visto
questi numeri, si'.

DUE SPECIFICITA'
-----------------
Un pixel di tessuto predetto GLASS non e' stato chiamato HSIL, quindi
entra nei veri negativi e GONFIA la specificita'. Con la soglia di
densita' ottica attuale questo non e' un caso di scuola: in
cross-validation 71,8M di pixel di tessuto annotato sono finiti nella
classe vetro, e in un fold la quota ha superato il 55% del tessuto
negativo. Si riportano quindi entrambe:

  specificita           : definizione standard, GLASS nei veri negativi
  specificita su tessuto: calcolata sui soli pixel predetti come
                          tessuto, insieme alla quota scartata

DUE MODALITA'
--------------
  ensemble  : media delle softmax dei modelli dei fold. Nessuno dei
              modelli ha mai visto le WSI di test, quindi e' lecito.
  singoli   : ogni modello valutato separatamente. Serve a mostrare che
              l'ensemble non e' un artificio e a quantificare la
              variabilita' fra MODELLI sullo stesso test set, che e'
              informazione diversa dalla variabilita' fra fold.

PREVALENZA
-----------
Precisione, Dice e IoU dipendono dalla prevalenza della classe
positiva; sensibilita' e specificita' no. Confrontare la precisione di
due partizioni con composizione diversa, o confrontare CV e test, senza
riportare la composizione non e' interpretabile. Lo script stampa la
composizione LAST del test accanto a quella della CV.

RILEVAZIONE A LIVELLO DI REGIONE
---------------------------------
La recall per pixel sottostima sistematicamente la capacita' di
segnalare una lesione, e su CIN1 lo fa per un motivo biologico: in una
regione annotata CIN1 il terzo superficiale e' maturo e citologicamente
quasi normale, ma porta comunque l'etichetta CIN1 su ogni pixel. Il
modello che li chiama Mucosa sta guardando qualcosa che E' mucosa
matura.

La domanda clinica non e' "quanti pixel", e' "la lesione viene
segnalata al patologo". Si misura per TILE: una tile che contiene una
lesione e' rilevata se una frazione sufficiente dei suoi pixel
lesionali viene predetta POSITIVA. E' l'equivalente della valutazione
lesion-level di CAMELYON16 (Ehteshami Bejnordi et al., JAMA 2017).
I tassi si mediano PER WSI e non sulle tile: le tile dentro un vetrino
sono correlate quanto i pixel, e una proporzione calcolata su tutte le
tile insieme e' pseudo-replicazione.

INTERVALLI DI CONFIDENZA
-------------------------
Bootstrap ricampionando le WSI, non i pixel: i pixel dentro una WSI
sono fortemente correlati (stesso paziente, stesso taglio, stessa
colorazione) e trattarli come indipendenti produrrebbe intervalli
assurdamente stretti. L'unita' statistica e' il vetrino.

Attenzione allo stimatore: per una metrica la cui stima puntuale e' una
MEDIA PER VETRINO (mIoU LAST per WSI, tassi di rilevazione) il
bootstrap deve mediare i valori per vetrino, non sommare le matrici di
confusione. Sommandole si ottiene l'intervallo di uno stimatore POOLED,
che e' un'altra quantita' ed e' dominata dai vetrini grandi.

Uso
----
    python test.py
====================================================================
"""

import os
import json
import math

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config as C
import dati as D
import modello as M
import metriche as MT
import grafici as GR
import esempi as ES

# ====================================================================
# PARAMETRI
# ====================================================================
NOME_REPORT = "report_test.json"

USA_TTA = True
N_BOOTSTRAP = 2000
SEED_BOOTSTRAP = 42

# Rilevazione per tile: una tile entra nel conteggio se almeno questa
# frazione dei suoi pixel appartiene alla lesione considerata.
FRAZIONE_MINIMA_LESIONE = 0.02
SOGLIE_RILEVAZIONE = (0.10, 0.25, 0.50)

# Ensemble aggiuntivo con i soli fold che hanno raggiunto l'operating
# point dichiarato in cross-validation.
ENSEMBLE_SOLO_AMMISSIBILI = True

# Esempi qualitativi in una cartella 'esempi'. La raccolta delle
# statistiche per tile avviene DURANTE il passaggio dell'ensemble
# argmax: non aggiunge inferenza, se non sulle poche tile selezionate.
SALVA_ESEMPI = True
# ====================================================================


def pulisci(o):
    """NaN e Inf non sono JSON valido: diventano null."""
    if isinstance(o, dict):
        return {k: pulisci(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [pulisci(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return pulisci(o.tolist())
    return o


# ====================================================================
# COERENZA FRA CV E TEST
# ====================================================================
def verifica_coerenza_run():
    """
    I checkpoint devono venire dallo STESSO run che ha prodotto il
    report di cross-validation nella stessa cartella.

    E' gia' successo di confrontare un report di CV e uno di test
    generati da configurazioni di loss diverse: le due matrici di
    confusione avevano firme d'errore opposte e ci e' voluto un giorno
    per capire perche'. Il controllo costa una lettura di file.

    Dalla versione con lo split vincolato si controlla anche il CSV di
    partizione: usare i checkpoint di un run con il CSV di un altro
    significa valutare su vetrini che erano in training, e il controllo
    di leakage a valle lo intercetterebbe comunque, ma tardi e con un
    messaggio meno chiaro.
    """
    path = C.path_metrica(C.NOME_REPORT)
    if not os.path.exists(path):
        print(f"   ATTENZIONE: {path} assente, impossibile verificare che i "
              f"checkpoint vengano da questa configurazione.")
        return None
    cv = json.load(open(path, encoding="utf-8"))
    cfg = cv.get("config", {})

    atteso = {"loss": C.parametri_loss(), "hed": C.parametri_hed(),
              "vetro": C.parametri_vetro()}
    diverse = [k for k, v in atteso.items()
               if k in cfg and json.dumps(cfg[k], sort_keys=True)
               != json.dumps(pulisci(v), sort_keys=True)]
    for chiave, atteso_val in (("csv_train", C.PATH_TRAIN_CSV),
                               ("csv_test", C.PATH_TEST_CSV)):
        if chiave in cfg and cfg[chiave] != atteso_val:
            diverse.append(f"{chiave} (CV: {cfg[chiave]})")
    if diverse:
        raise RuntimeError(
            f"I checkpoint in {C.DIR_MODELLI} sono stati addestrati con una "
            f"configurazione diversa da quella attuale (sezioni: "
            f"{', '.join(diverse)}). Ripristinare config.py oppure "
            f"riaddestrare prima di valutare il test.")
    print(f"   configurazione coerente col report di CV "
          f"(run '{cfg.get('nome_run', '?')}', "
          f"architettura {cfg.get('architettura', '?')})")
    if "csv_test" not in cfg:
        print("   NOTA: il report di CV non registra i CSV di split. "
              "Aggiungerli in esegui.py per i run futuri.")
    return cv


def fold_ammissibili(cv):
    """
    Quali fold hanno raggiunto l'operating point dichiarato.

    Un fold di ripiego e' selezionato su mIoU e non sul vincolo: il suo
    modello sta in un punto di lavoro diverso dagli altri. Va sempre
    dichiarato, perche' un ensemble che mescola criteri di selezione
    diversi non e' descrivibile in una riga di metodi.
    """
    if not cv or "cv" not in cv:
        return None
    stato = {}
    for r in cv["cv"]:
        sel = r.get("selezione", {})
        stato[int(r["fold"])] = bool(sel.get("ammissibile", True))
    return stato


def confronta_con_cv(cv, ris_test):
    """Il divario CV -> test e' il numero che interessa alla tesi."""
    if not cv:
        return None
    agg = cv.get("aggregati", {})
    h = cv.get("hsil_aggregato", {})
    lh = cv.get("lsil_vs_hsil_aggregato", {})
    diag = cv.get("diagnosi", {})
    righe = [
        ("sensibilita HSIL", h.get("sensibilita"),
         ris_test["hsil"]["sensibilita"]),
        ("specificita HSIL", h.get("specificita"),
         ris_test["hsil"]["specificita"]),
        ("precisione HSIL", h.get("precisione"),
         ris_test["hsil"]["precisione"]),
        ("recall LSIL (vs HSIL)", lh.get("recall_lsil"),
         ris_test["lsil_vs_hsil"]["recall_lsil"]),
        ("accuratezza bilanciata", cv.get("accbil_last_aggregata"),
         ris_test["accbil_last"]),
        ("mIoU LAST per WSI", (agg.get("miou_last_per_wsi") or {}).get("media"),
         ris_test["miou_last_per_wsi"]),
    ]
    print("\n" + "=" * 78)
    print("  DIVARIO CROSS-VALIDATION -> TEST")
    print("=" * 78)
    print(f"  {'metrica':<26}{'CV':>10}{'test':>10}{'delta':>10}")
    print("  " + "-" * 56)
    out = {}
    for nome, a, b in righe:
        if a is None or b is None or np.isnan(a) or np.isnan(b):
            continue
        out[nome] = {"cv": float(a), "test": float(b), "delta": float(b - a)}
        print(f"  {nome:<26}{a:>10.4f}{b:>10.4f}{b-a:>+10.4f}")

    n_co = diag.get("n_wsi_lsil_e_hsil", "?")
    n_wsi = diag.get("n_wsi", "?")
    print(f"\n  ATTENZIONE alla precisione: dipende dalla prevalenza, che nel")
    print("  test e' diversa da quella della CV (vedi tabella COMPOSIZIONE).")
    print("  Un delta positivo sulla precisione NON e' di per se' un")
    print("  miglioramento del modello.")
    print(f"\n  Un divario negativo ampio su LSIL e' atteso e va spiegato, non")
    print(f"  nascosto: CIN1 e HSIL coesistono in {n_co} WSI su {n_wsi}, quindi")
    print("  in training il grado e' quasi perfettamente predetto dall'identita'")
    print("  del vetrino (Howard et al. 2021, Nat Commun 12:4423). Il test")
    print("  misura quanto il modello si appoggia a quell'indizio.")
    return out


# ====================================================================
# MODELLI E INFERENZA
# ====================================================================
def carica_modelli(device, ammissibili=None):
    """-> (modelli, nomi, indici_ammissibili)"""
    modelli, nomi, idx_amm = [], [], []
    for k in range(C.N_FOLD):
        path = C.path_modello(k)
        if not os.path.exists(path):
            print(f"   fold {k}: checkpoint mancante, salto ({path})")
            continue
        m = M.build_model(device, verbose=False)
        m.load_state_dict(torch.load(path, map_location=device,
                                     weights_only=True))
        m.eval()
        if ammissibili is not None and not ammissibili.get(k, True):
            nomi.append(f"fold {k} (RIPIEGO)")
        else:
            nomi.append(f"fold {k}")
            idx_amm.append(len(modelli))
        modelli.append(m)
    if not modelli:
        raise RuntimeError(f"Nessun checkpoint trovato in {C.DIR_MODELLI}")
    if len(modelli) < C.N_FOLD:
        print(f"   ATTENZIONE: {C.N_FOLD - len(modelli)} checkpoint mancanti. "
              f"L'ensemble e' piu' piccolo del previsto.")
    print(f"   modelli caricati: {len(modelli)}  ({', '.join(nomi)})")
    n_rip = len(modelli) - len(idx_amm)
    if n_rip:
        print(f"   {n_rip} fold su {len(modelli)} sono di RIPIEGO: selezionati")
        print("   su mIoU perche' nessuna epoca ha rispettato l'operating")
        print("   point. Sono in un punto di lavoro diverso dagli altri.")
    return modelli, nomi, idx_amm


@torch.no_grad()
def _softmax_batch(modelli, images, amp, tta=USA_TTA):
    """
    Media delle softmax sui modelli e sulle varianti TTA.

    Si mediano le PROBABILITA' e non i logit: i logit di modelli diversi
    non sono su scale confrontabili, mentre le probabilita' sono tutte
    normalizzate a 1 e la media resta una distribuzione valida.
    """
    somma, n = None, 0
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


def decidi(prob, omega=None):
    """
    omega None -> argmax, cioe' minimo errore 0-1.
    omega dato -> minimo rischio atteso: y = argmin_j sum_i p_i Omega[i,j].
    """
    if omega is None:
        return prob.argmax(1)
    return torch.einsum("bihw,ij->bjhw", prob, omega).argmin(1)


# ====================================================================
# METRICHE AGGIUNTIVE
# ====================================================================
def specificita_su_tessuto(cm_last):
    """
    Specificita' HSIL calcolata sui soli pixel di tessuto negativo che
    il modello ha effettivamente classificato come TESSUTO.

    Nella definizione standard un pixel di stroma predetto GLASS non e'
    stato chiamato HSIL e quindi conta come vero negativo: la
    specificita' sale perche' il modello ha cancellato il tessuto, non
    perche' lo ha classificato bene. Va riportata insieme alla quota
    scartata, altrimenti il numero e' fuorviante.

    -> (specificita_condizionata, quota di tessuto negativo -> GLASS)
    """
    cm = cm_last.double()
    non_hsil = [C.IDX_NEGATIVE_LAST, C.IDX_LSIL_LAST]
    tessuto = [C.IDX_NEGATIVE_LAST, C.IDX_LSIL_LAST, C.IDX_HSIL_LAST]
    tot = float(cm[non_hsil, :].sum())
    verso_vetro = float(cm[non_hsil, C.IDX_GLASS_LAST].sum())
    fp = float(cm[non_hsil, C.IDX_HSIL_LAST].sum())
    predetti_tessuto = float(cm[non_hsil][:, tessuto].sum())
    tn = predetti_tessuto - fp
    spec = tn / predetti_tessuto if predetti_tessuto > 0 else float("nan")
    quota = verso_vetro / tot if tot > 0 else float("nan")
    return spec, quota


def composizione_last(cm_last):
    """Quote di NEGATIVE / LSIL / HSIL sui pixel di tessuto annotato."""
    n = cm_last.double().sum(1)[1:]
    tot = float(n.sum())
    return {C.NOMI_LAST[i + 1]: (float(n[i]), float(n[i]) / tot if tot else 0.0)
            for i in range(3)}


def stampa_composizione(cm_last, cv):
    """
    Precisione, Dice e IoU dipendono dalla prevalenza; sensibilita' e
    specificita' no. Senza questa tabella accanto alle metriche, il
    confronto fra CV e test (o fra due split) non e' interpretabile.
    """
    comp = composizione_last(cm_last)
    print("\n  COMPOSIZIONE DEL TEST (pixel di tessuto annotato)")
    print(f"    {'classe':<10}{'quota':>9}{'pixel':>18}")
    for nome, (n, q) in comp.items():
        print(f"    {nome:<10}{100*q:>8.1f}%{int(n):>18,}")
    if cv and "cm_last_aggregata" in cv:
        comp_cv = composizione_last(torch.tensor(cv["cm_last_aggregata"]))
        riga = "  ".join(f"{k} {100*v[1]:.1f}%" for k, v in comp_cv.items())
        print(f"    in cross-validation: {riga}")
        delta = max(abs(comp[k][1] - comp_cv[k][1]) for k in comp)
        if delta > 0.05:
            print(f"    ATTENZIONE: scarto massimo di prevalenza "
                  f"{100*delta:.1f} punti. Precisione, Dice e IoU non sono")
            print("    confrontabili con la CV senza tenerne conto.")
    return comp


# ====================================================================
# RILEVAZIONE A LIVELLO DI TILE
# ====================================================================
class Rilevazione:
    """
    Per ogni tile e per ogni classe lesionale presente sopra un'area
    minima, registra la frazione dei suoi pixel predetta POSITIVA
    (LSIL o HSIL) e la frazione predetta con il grado ESATTO.

    La prima risponde a "la lesione viene segnalata", la seconda a "il
    grado e' corretto". Sono due domande cliniche diverse: la prima
    decide se il vetrino finisce sotto gli occhi del patologo, la
    seconda se la gestione ASCCP e' quella giusta.

    I tassi si aggregano PER WSI. Le tile dentro un vetrino condividono
    paziente, taglio e colorazione: una proporzione calcolata su tutte
    le tile insieme e' pseudo-replicazione, esattamente l'errore che il
    bootstrap sui vetrini esiste per evitare.
    """

    def __init__(self, device):
        self.lut = C.LUT_LAST.to(device)
        self.dati = {C.IDX_LSIL_LAST: [], C.IDX_HSIL_LAST: []}

    def aggiorna(self, pred, masks, wsis):
        valido = masks < C.NUM_CLASSES
        m_safe = torch.where(valido, masks, torch.zeros_like(masks))
        t_last = torch.where(valido, self.lut[m_safe],
                             torch.full_like(masks, 255))
        p_last = self.lut[pred]
        positivo = ((p_last == C.IDX_LSIL_LAST) | (p_last == C.IDX_HSIL_LAST))

        n_px = masks.shape[1] * masks.shape[2]
        for c in self.dati:
            m = (t_last == c)
            n_vera = m.flatten(1).sum(1)
            abbastanza = n_vera >= FRAZIONE_MINIMA_LESIONE * n_px
            if not bool(abbastanza.any()):
                continue
            n_pos = (m & positivo).flatten(1).sum(1)
            n_exa = (m & (p_last == c)).flatten(1).sum(1)
            for b in torch.nonzero(abbastanza).flatten().tolist():
                self.dati[c].append((wsis[b],
                                     float(n_pos[b] / n_vera[b]),
                                     float(n_exa[b] / n_vera[b])))

    @staticmethod
    def _per_wsi(voci, soglia, campo):
        """Tasso di rilevazione per vetrino -> {wsi: tasso}."""
        agg = {}
        for w, pos, exa in voci:
            v = pos if campo == "pos" else exa
            agg.setdefault(w, []).append(1.0 if v >= soglia else 0.0)
        return {w: float(np.mean(v)) for w, v in agg.items()}

    def _boot(self, per_wsi, n=N_BOOTSTRAP, seed=SEED_BOOTSTRAP):
        nomi = sorted(per_wsi)
        if len(nomi) < 3:
            return None
        rng = np.random.default_rng(seed)
        v = [float(np.mean([per_wsi[nomi[i]] for i in
                            rng.choice(len(nomi), len(nomi), replace=True)]))
             for _ in range(n)]
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    def riassumi(self):
        out = {}
        print("\n  RILEVAZIONE PER TILE (tassi mediati per WSI)")
        print("  Una tile e' 'rilevata' se almeno la quota indicata dei suoi")
        print("  pixel lesionali e' predetta POSITIVA (LSIL o HSIL).")
        print(f"  Sono incluse le tile con almeno il "
              f"{100*FRAZIONE_MINIMA_LESIONE:.0f}% di pixel della lesione.")
        intest = "".join(f"{'>=' + str(int(100*s)) + '%':>9}"
                         for s in SOGLIE_RILEVAZIONE)
        print(f"\n  {'classe':<8}{'n tile':>8}{'n WSI':>7}{intest}"
              f"{'grado ok':>10}")
        print("  " + "-" * 62)
        for c, nome in ((C.IDX_LSIL_LAST, "LSIL"), (C.IDX_HSIL_LAST, "HSIL")):
            voci = self.dati[c]
            if not voci:
                print(f"  {nome:<8}{0:>8}   (nessuna tile sopra l'area minima)")
                out[nome] = None
                continue
            wsi_set = {w for w, _, _ in voci}
            tassi, ic = {}, {}
            for s in SOGLIE_RILEVAZIONE:
                pw = self._per_wsi(voci, s, "pos")
                tassi[f"{s:.2f}"] = float(np.mean(list(pw.values())))
                ic[f"{s:.2f}"] = self._boot(pw)
            pw_ex = self._per_wsi(voci, 0.25, "exa")
            grado = float(np.mean(list(pw_ex.values())))
            riga = "".join(f"{t:>9.3f}" for t in tassi.values())
            print(f"  {nome:<8}{len(voci):>8}{len(wsi_set):>7}{riga}"
                  f"{grado:>10.3f}")
            out[nome] = {
                "n_tile": len(voci),
                "n_wsi": len(wsi_set),
                "rilevazione_media_per_wsi": tassi,
                "ic95": ic,
                "per_wsi": {s: self._per_wsi(voci, float(s), "pos")
                            for s in tassi},
                "grado_esatto_25": grado,
                "grado_esatto_25_per_wsi": pw_ex,
            }
        print("\n  L'ultima colonna e' la quota di tile in cui almeno il 25% dei")
        print("  pixel riceve il grado CORRETTO, non solo 'positivo'.")
        for nome, v in out.items():
            if v and v["ic95"].get("0.25"):
                lo, hi = v["ic95"]["0.25"]
                print(f"  {nome} a soglia 25%: "
                      f"{v['rilevazione_media_per_wsi']['0.25']:.3f} "
                      f"IC95% [{lo:.3f}, {hi:.3f}]  (n = {v['n_wsi']} WSI)")
        return out


# ====================================================================
# VALUTAZIONE
# ====================================================================
@torch.no_grad()
def valuta(modelli, loader, device, amp, etichetta="", omega=None,
           con_rilevazione=False, raccolta=None):
    """
    -> (cm_fine, cm_tile, {wsi: cm_fine}, rilevazione|None)

    'raccolta' e' un esempi.RaccoglitoreTile opzionale: registra la
    matrice di confusione di ogni singola tile mentre le predizioni
    scorrono, cosi' la scelta degli esempi qualitativi non richiede un
    secondo passaggio sul test set.
    """
    cm = MT.nuova_cm(device=device)
    cm_tile = MT.nuova_cm(device=device)
    cm_wsi = {}
    ril = Rilevazione(device) if con_rilevazione else None

    desc = f"  {etichetta}" if etichetta else "  test"
    for b_idx, (images, masks, wsis) in enumerate(
            tqdm(loader, desc=desc, ncols=78)):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        pred = decidi(_softmax_batch(modelli, images, amp), omega)
        MT.aggiorna_cm(cm, pred, masks)
        MT.aggiorna_cm_tile(cm_tile, pred, masks)
        if raccolta is not None:
            # shuffle=False e nessun sampler: la posizione nel loader
            # e' l'indice in campioni. registra() lo verifica.
            raccolta.registra(b_idx * loader.batch_size, pred, masks,
                              list(wsis))
        if ril is not None:
            ril.aggiorna(pred, masks, list(wsis))
        for w in dict.fromkeys(wsis):
            sel = [j for j, x in enumerate(wsis) if x == w]
            cm_wsi.setdefault(w, MT.nuova_cm(device=device))
            MT.aggiorna_cm(cm_wsi[w], pred[sel], masks[sel])
    return (cm.cpu(), cm_tile.cpu(),
            {w: c.cpu() for w, c in cm_wsi.items()},
            ril.riassumi() if ril is not None else None)


def bootstrap_wsi_pooled(cm_wsi, funzione, n=N_BOOTSTRAP, seed=SEED_BOOTSTRAP):
    """
    IC95% per uno stimatore POOLED: si ricampionano le WSI, si SOMMANO
    le loro matrici e si calcola la metrica sulla matrice risultante.

    Da usare per sensibilita', specificita', precisione e accuratezza
    bilanciata, la cui stima puntuale e' calcolata sulla matrice
    aggregata di tutto il test.
    """
    nomi = sorted(cm_wsi.keys())
    if len(nomi) < 3:
        return None
    rng = np.random.default_rng(seed)
    valori = []
    for _ in range(n):
        scelte = rng.choice(len(nomi), len(nomi), replace=True)
        v = funzione(sum(cm_wsi[nomi[i]] for i in scelte))
        if v is not None and not np.isnan(v):
            valori.append(v)
    if not valori:
        return None
    return (float(np.percentile(valori, 2.5)),
            float(np.percentile(valori, 97.5)))


def bootstrap_wsi_media(cm_wsi, funzione, n=N_BOOTSTRAP, seed=SEED_BOOTSTRAP):
    """
    IC95% per uno stimatore che e' una MEDIA PER VETRINO: si calcola la
    metrica su ogni WSI separatamente e si media sui vetrini estratti.

    Da usare per la mIoU LAST per WSI. Usare bootstrap_wsi_pooled per
    quella metrica darebbe l'intervallo di una mIoU pooled, che e'
    un'altra quantita' ed e' dominata dai vetrini grandi: la stima
    puntuale e l'intervallo si riferirebbero a stimatori diversi.
    """
    nomi = sorted(cm_wsi.keys())
    if len(nomi) < 3:
        return None
    per_wsi = {}
    for w in nomi:
        v = funzione(cm_wsi[w])
        if v is not None and not np.isnan(v):
            per_wsi[w] = float(v)
    nomi = sorted(per_wsi)
    if len(nomi) < 3:
        return None
    rng = np.random.default_rng(seed)
    valori = [float(np.mean([per_wsi[nomi[i]] for i in
                             rng.choice(len(nomi), len(nomi), replace=True)]))
              for _ in range(n)]
    return (float(np.percentile(valori, 2.5)),
            float(np.percentile(valori, 97.5)))


def jackknife_wsi(cm_wsi, funzione):
    """
    Escludendo una WSI per volta: con pochi vetrini dice quanto il
    risultato dipende da un singolo caso, cosa che il bootstrap con n
    cosi' piccolo comunica male.
    """
    nomi = sorted(cm_wsi.keys())
    if len(nomi) < 3:
        return None
    fuori = {}
    for w in nomi:
        v = funzione(sum(cm_wsi[x] for x in nomi if x != w))
        if v is not None and not np.isnan(v):
            fuori[w] = float(v)
    return fuori or None


# ====================================================================
# RIEPILOGO
# ====================================================================
def riassumi(cm, cm_tile, cm_wsi, titolo, rilevazione=None, cv=None):
    cm_last = MT.cm_to_last(cm)
    hsil = MT.metriche_hsil(cm_last)
    lh = MT.discriminazione_lsil_hsil(cm_last)
    spec_tess, quota_vetro = specificita_su_tessuto(cm_last)

    print("\n" + "=" * 78)
    print(f"  {titolo}")
    print("=" * 78)
    MT.stampa_confusione(cm, C.CLASS_NAMES,
                         "confusione classi fini (% per riga)")
    print()
    MT.stampa_per_classe(cm, C.CLASS_NAMES)
    print()
    MT.stampa_confusione(cm_last, C.NOMI_LAST, "confusione LAST (% per riga)")
    print()
    MT.stampa_errori_principali(cm, C.CLASS_NAMES, top=6)

    comp = stampa_composizione(cm_last, cv)

    f_sens = lambda c: MT.metriche_hsil(MT.cm_to_last(c))["sensibilita"]
    f_spec = lambda c: MT.metriche_hsil(MT.cm_to_last(c))["specificita"]
    f_prec = lambda c: MT.metriche_hsil(MT.cm_to_last(c))["precisione"]
    f_spet = lambda c: specificita_su_tessuto(MT.cm_to_last(c))[0]
    f_lsil = lambda c: MT.discriminazione_lsil_hsil(MT.cm_to_last(c))["recall_lsil"]
    f_miou = lambda c: MT.macro_presenti(MT.cm_to_last(c),
                                         escludi=(C.IDX_GLASS_LAST,))
    f_accb = lambda c: MT.accuratezza_bilanciata(MT.cm_to_last(c),
                                                 escludi=(C.IDX_GLASS_LAST,))
    # pooled: la stima puntuale e' calcolata sulla matrice aggregata
    ic = {k: bootstrap_wsi_pooled(cm_wsi, f) for k, f in
          (("sensibilita", f_sens), ("specificita", f_spec),
           ("precisione", f_prec), ("specificita_su_tessuto", f_spet),
           ("recall_lsil", f_lsil), ("accbil", f_accb))}
    # media per vetrino: stimatore diverso, bootstrap diverso
    ic["miou"] = bootstrap_wsi_media(cm_wsi, f_miou)

    def _ic(t):
        return f"  IC95% [{t[0]:.3f}, {t[1]:.3f}]" if t else ""

    print(f"\n  DELIVERABLE PRIMARIO — HSIL vs non-HSIL")
    print(f"    sensibilita  {hsil['sensibilita']:.4f}{_ic(ic['sensibilita'])}")
    print(f"    specificita  {hsil['specificita']:.4f}{_ic(ic['specificita'])}")
    print(f"    precisione   {hsil['precisione']:.4f}{_ic(ic['precisione'])}")
    print(f"    Dice         {hsil['dice']:.4f}")
    print(f"\n    specificita sui soli pixel predetti TESSUTO: "
          f"{spec_tess:.4f}{_ic(ic['specificita_su_tessuto'])}")
    print(f"    tessuto non-HSIL predetto vetro: {100*quota_vetro:.1f}%  "
          f"({int(hsil['px_tessuto_predetti_vetro']):,} px in totale)")
    if quota_vetro > 0.05:
        print("    Con questa quota la specificita' standard e' gonfiata: i")
        print("    pixel cancellati contano come veri negativi. Riportare")
        print("    entrambe le cifre.")
    soglia = C.SENS_HSIL_MINIMA
    stato = "RISPETTATO" if hsil["sensibilita"] >= soglia else "NON rispettato"
    print(f"\n    operating point dichiarato in CV (sens >= {soglia:.2f}): "
          f"{stato}")

    accbil = MT.accuratezza_bilanciata(cm_last, escludi=(C.IDX_GLASS_LAST,))
    print(f"\n  accuratezza bilanciata LAST (GLASS escluso): "
          f"{accbil:.4f}{_ic(ic['accbil'])}")
    print(f"  accuratezza per tile: {MT.accuratezza(cm_tile):.4f}"
          f"   bilanciata: {MT.accuratezza_bilanciata(cm_tile):.4f}")
    n_co = ((cv or {}).get("diagnosi", {}) or {}).get("n_wsi_lsil_e_hsil", "?")
    print(f"\n  Confine LSIL/HSIL (esplorativo, {n_co} WSI di co-occorrenza in CV):")
    print(f"    accuratezza {lh['accuratezza']:.4f}   "
          f"recall LSIL {lh['recall_lsil']:.4f}{_ic(ic['recall_lsil'])}   "
          f"recall HSIL {lh['recall_hsil']:.4f}")

    # ---- tabella per WSI ----
    per_wsi, per_wsi_dett = {}, {}
    print(f"\n  Per WSI (mIoU LAST sulle classi presenti){_ic(ic['miou'])}:")
    print(f"    {'WSI':<14}{'mIoU':>7}{'sens':>7}{'spec':>7}{'specT':>7}"
          f"{'->vetro':>9}   recall per classe")
    for w, c in sorted(cm_wsi.items(),
                       key=lambda kv: MT.macro_presenti(
                           MT.cm_to_last(kv[1]), escludi=(C.IDX_GLASS_LAST,))):
        cl = MT.cm_to_last(c)
        v = MT.macro_presenti(cl, escludi=(C.IDX_GLASS_LAST,))
        h_w = MT.metriche_hsil(cl)
        st_w, qv_w = specificita_su_tessuto(cl)
        per_wsi[w] = float(v)
        rec = MT.recall_per_classe(cl)
        n = cl.double().sum(1)
        dett = "  ".join(
            f"{C.NOMI_LAST[i]} {100*rec[i]:.0f}%" if n[i] > 0
            else f"{C.NOMI_LAST[i]} -" for i in range(1, 4))
        per_wsi_dett[w] = {
            "miou": float(v),
            "sensibilita": h_w["sensibilita"],
            "specificita": h_w["specificita"],
            "specificita_su_tessuto": float(st_w),
            "quota_tessuto_vetro": float(qv_w),
            "recall_last": {C.NOMI_LAST[i]: rec[i] for i in range(1, 4)},
        }

        def _f(x):
            return "  -  " if x is None or np.isnan(x) else f"{x:.3f}"
        print(f"    {w:<14}{v:>7.3f}{_f(h_w['sensibilita']):>7}"
              f"{_f(h_w['specificita']):>7}{_f(st_w):>7}"
              f"{100*qv_w:>8.1f}%   {dett}")
    vals = list(per_wsi.values())
    print(f"    media {np.mean(vals):.4f}   mediana {np.median(vals):.4f}   "
          f"std {np.std(vals):.4f}   (n = {len(vals)} WSI)")
    print("    specT = specificita sui soli pixel predetti tessuto;")
    print("    ->vetro = quota di tessuto non-HSIL classificato come vetro.")

    jk = jackknife_wsi(cm_wsi, f_sens)
    if jk:
        peggio = min(jk, key=jk.get)
        print(f"\n  Sensibilita' HSIL escludendo una WSI per volta: "
              f"min {min(jk.values()):.4f} (senza {peggio}), "
              f"max {max(jk.values()):.4f}")
        print(f"  Con {len(vals)} vetrini l'intervallo bootstrap e' dominato da")
        print("  QUALI vetrini vengono estratti. Riportarlo insieme all'elenco")
        print("  per WSI qui sopra, mai da solo.")

    return {
        "hsil": hsil,
        "specificita_su_tessuto": float(spec_tess),
        "quota_tessuto_predetto_vetro": float(quota_vetro),
        "composizione_last": {k: {"pixel": v[0], "quota": v[1]}
                              for k, v in comp.items()},
        "lsil_vs_hsil": lh,
        "accbil_last": float(accbil),
        "accbil_tile": float(MT.accuratezza_bilanciata(cm_tile)),
        "miou_last_per_wsi": float(np.mean(vals)),
        "ic95": ic,
        "jackknife_sensibilita": jk,
        "rilevazione_per_tile": rilevazione,
        "iou_fine": [None if np.isnan(v) else float(v)
                     for v in MT.iou_per_classe(cm).tolist()],
        "recall_fine": MT.recall_per_classe(cm),
        "cm_fine": cm.tolist(),
        "cm_last": cm_last.tolist(),
        "cm_tile": cm_tile.tolist(),
        "per_wsi": per_wsi,
        "per_wsi_dettaglio": per_wsi_dett,
    }


# ====================================================================
# MAIN
# ====================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Run: {C.NOME_RUN}   |   {C.firma_esperimento()}")
    print(f"Split: {os.path.basename(C.PATH_TEST_CSV)}")
    print("\n" + "!" * 78)
    print("  Questa valutazione va eseguita UNA VOLTA SOLA.")
    print("  Rilanciarla dopo aver cambiato qualcosa in base ai suoi risultati")
    print("  trasforma il test set in un secondo validation set.")
    print("!" * 78)

    print("\n--- COERENZA ---")
    cv = verifica_coerenza_run()
    ammissibili = fold_ammissibili(cv)

    print("\n--- DATI DI TEST ---")
    campioni_tot = D.elenca_campioni()
    df = pd.read_csv(C.PATH_TEST_CSV, index_col=False, encoding="utf-8")
    wsi_test = set(df["WSI"].tolist())
    campioni = [c for c in campioni_tot if c[2] in wsi_test]
    if not campioni:
        print(f"Nessuna tile di test trovata. Verificare {C.PATH_TEST_CSV}")
        return

    wsi_trovate = sorted({c[2] for c in campioni})
    print(f"   WSI di test attese : {len(wsi_test)}")
    print(f"   WSI di test trovate: {len(wsi_trovate)}  "
          f"({', '.join(wsi_trovate)})")
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
    modelli, nomi, idx_amm = carica_modelli(device, ammissibili)
    amp = (device.type == "cuda")
    omega = (torch.tensor(C.MATRICE_COSTI, dtype=torch.float32, device=device)
             if C.MATRICE_COSTI is not None else None)

    # ---------------- ensemble, decisione argmax ----------------
    racc = ES.RaccoglitoreTile(campioni) if SALVA_ESEMPI else None
    cm, cm_tile, cm_wsi, ril = valuta(
        modelli, loader, device, amp, f"ensemble argmax ({len(modelli)} mod.)",
        omega=None, con_rilevazione=True, raccolta=racc)
    ris_ens = riassumi(cm, cm_tile, cm_wsi,
                       f"TEST — ENSEMBLE DI {len(modelli)} MODELLI"
                       f"{' + TTA' if USA_TTA else ''} — decisione ARGMAX",
                       rilevazione=ril, cv=cv)

    # ---------------- esempi qualitativi ----------------
    # Vengono dall'ENSEMBLE con ARGMAX, cioe' dal sistema di cui si
    # riportano le metriche principali. Basarli sul 'fold migliore'
    # sarebbe selezione sui punteggi di validation, e le figure
    # illustrerebbero un sistema diverso da quello misurato.
    manifest_esempi = None
    if SALVA_ESEMPI:
        manifest_esempi = ES.salva_esempi(
            racc, cm, modelli, device, amp, _softmax_batch, omega=omega,
            etichetta_sistema=f"ensemble di {len(modelli)} modelli"
                              f"{' + TTA' if USA_TTA else ''}, argmax")

    # ---------------- ensemble, decisione a costo minimo ----------------
    ris_cost = None
    if omega is not None:
        cmc, cmc_tile, cmc_wsi, rilc = valuta(
            modelli, loader, device, amp, "ensemble costo minimo",
            omega=omega, con_rilevazione=True)
        ris_cost = riassumi(cmc, cmc_tile, cmc_wsi,
                            "TEST — STESSO ENSEMBLE — decisione a COSTO MINIMO "
                            "(Omega prespecificata)", rilevazione=rilc, cv=cv)
        print("\n  Le due decisioni usano le STESSE probabilita': la differenza")
        print("  e' solo la regola di scelta. Omega e' quella dichiarata in")
        print("  config e gia' usata nella loss, non e' stata stimata qui.")

    # ---------------- ensemble dei soli fold ammissibili ----------------
    ris_amm = None
    if (ENSEMBLE_SOLO_AMMISSIBILI and idx_amm
            and 0 < len(idx_amm) < len(modelli)):
        sub = [modelli[i] for i in idx_amm]
        cma, cma_tile, cma_wsi, _ = valuta(
            sub, loader, device, amp,
            f"ensemble solo ammissibili ({len(sub)} mod.)", omega=None)
        ris_amm = riassumi(
            cma, cma_tile, cma_wsi,
            f"TEST — ENSEMBLE DEI SOLI {len(sub)} FOLD AMMISSIBILI "
            f"(analisi di sensibilita')", cv=cv)
        print("\n  Questo ensemble esclude i fold selezionati col criterio di")
        print("  ripiego. Non e' il risultato principale: serve a mostrare")
        print("  quanto la loro inclusione sposta le metriche.")
        d = (ris_amm["hsil"]["sensibilita"] - ris_ens["hsil"]["sensibilita"],
             ris_amm["hsil"]["specificita"] - ris_ens["hsil"]["specificita"],
             ris_amm["miou_last_per_wsi"] - ris_ens["miou_last_per_wsi"])
        print(f"  delta rispetto all'ensemble completo: sens {d[0]:+.4f}   "
              f"spec {d[1]:+.4f}   mIoU {d[2]:+.4f}")

    # ---------------- modelli singoli ----------------
    print("\n" + "=" * 78)
    print("  TEST — MODELLI SINGOLI (stesso test set, decisione argmax)")
    print("=" * 78)
    print(f"  {'modello':<18}{'sensHSIL':>10}{'specHSIL':>10}"
          f"{'recLSIL':>10}{'accBil':>9}{'mIoULAST':>10}")
    print("  " + "-" * 68)
    singoli = {}
    for m, nome in zip(modelli, nomi):
        c, ct, cw, _ = valuta([m], loader, device, amp, nome)
        cl = MT.cm_to_last(c)
        h = MT.metriche_hsil(cl)
        l = MT.discriminazione_lsil_hsil(cl)
        vals = [MT.macro_presenti(MT.cm_to_last(x), escludi=(C.IDX_GLASS_LAST,))
                for x in cw.values()]
        vals = [v for v in vals if not np.isnan(v)]
        ab = MT.accuratezza_bilanciata(cl, escludi=(C.IDX_GLASS_LAST,))
        singoli[nome] = {"sensibilita": h["sensibilita"],
                         "specificita": h["specificita"],
                         "recall_lsil": l["recall_lsil"],
                         "accbil_last": float(ab),
                         "miou_last_per_wsi": float(np.mean(vals)) if vals else None}
        print(f"  {nome:<18}{h['sensibilita']:>10.4f}{h['specificita']:>10.4f}"
              f"{l['recall_lsil']:>10.4f}{ab:>9.4f}{np.mean(vals):>10.4f}")

    ms = float(np.mean([v["sensibilita"] for v in singoli.values()]))
    print(f"\n  media dei singoli: {ms:.4f}   ensemble: "
          f"{ris_ens['hsil']['sensibilita']:.4f}   "
          f"({ris_ens['hsil']['sensibilita'] - ms:+.4f})")
    print("  La variabilita' qui e' fra MODELLI sullo stesso test set: e'")
    print("  informazione diversa dalla variabilita' fra fold, che mescolava")
    print("  modelli diversi E insiemi di validation diversi.")

    divario = confronta_con_cv(cv, ris_ens)

    report = {
        "run": C.NOME_RUN,
        "firma": C.firma_esperimento(),
        "config": {
            "architettura": type(modelli[0]).__name__,
            "encoder": C.ENCODER,
            "tile": C.TILE,
            "csv_train": C.PATH_TRAIN_CSV,
            "csv_test": C.PATH_TEST_CSV,
            "hed": C.parametri_hed(),
            "vetro": C.parametri_vetro(),
            "loss": C.parametri_loss(),
            "selezione": C.parametri_selezione(),
        },
        "csv_test": C.PATH_TEST_CSV,
        "wsi_test": wsi_trovate,
        "n_tile": len(campioni),
        "n_modelli": len(modelli),
        "nomi_modelli": nomi,
        "fold_ammissibili": ammissibili,
        "tta": USA_TTA,
        "n_bootstrap": N_BOOTSTRAP,
        "frazione_minima_lesione": FRAZIONE_MINIMA_LESIONE,
        "ensemble_argmax": ris_ens,
        "ensemble_costo_minimo": ris_cost,
        "ensemble_solo_ammissibili": ris_amm,
        "singoli": singoli,
        "divario_cv_test": divario,
        "esempi_qualitativi": (
            {"cartella": ES.NOME_CARTELLA,
             "n_esempi": len(manifest_esempi["esempi"]),
             "regola": manifest_esempi["regola"]}
            if manifest_esempi else None),
    }
    with open(C.path_metrica(NOME_REPORT), "w", encoding="utf-8") as f:
        json.dump(pulisci(report), f, indent=2, ensure_ascii=False)
    print(f"\nReport di test: {C.path_metrica(NOME_REPORT)}")

    # ---------------- grafici ----------------
    print("\n--- GRAFICI DI TEST ---")
    percorsi = {
        "confusioni_test": GR.plot_confusioni(
            torch.tensor(ris_ens["cm_fine"]),
            torch.tensor(ris_ens["cm_last"]),
            torch.tensor(ris_ens["cm_tile"]),
            nome_file="confusioni_test_ensemble.png",
            sottotitolo=f"Test — ensemble di {len(modelli)} modelli (argmax)"),
        "miou_per_wsi_test": GR.plot_per_wsi(
            ris_ens["per_wsi"], nome_file="miou_per_wsi_test_ensemble.png"),
    }
    if ris_cost is not None:
        percorsi["confusioni_test_costo"] = GR.plot_confusioni(
            torch.tensor(ris_cost["cm_fine"]),
            torch.tensor(ris_cost["cm_last"]),
            torch.tensor(ris_cost["cm_tile"]),
            nome_file="confusioni_test_costo_minimo.png",
            sottotitolo="Test — decisione a costo minimo")
    if ris_amm is not None:
        percorsi["confusioni_test_ammissibili"] = GR.plot_confusioni(
            torch.tensor(ris_amm["cm_fine"]),
            torch.tensor(ris_amm["cm_last"]),
            torch.tensor(ris_amm["cm_tile"]),
            nome_file="confusioni_test_solo_ammissibili.png",
            sottotitolo="Test — soli fold ammissibili")

    GR.salva_csv(np.array(ris_ens["cm_fine"]), C.CLASS_NAMES,
                 "confusione_fine_test.csv")
    GR.salva_csv(np.array(ris_ens["cm_last"]), C.NOMI_LAST,
                 "confusione_last_test.csv")
    GR.salva_csv(np.array(ris_ens["cm_tile"]), C.CLASS_NAMES,
                 "confusione_tile_test.csv")

    for k, p in percorsi.items():
        if p:
            print(f"    {k:<30}{p}")


if __name__ == "__main__":
    main()