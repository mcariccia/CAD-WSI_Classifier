"""
confronto_prima_dopo.py
====================================================================
Stabilisce se lo sbiadimento dei tile viene dal VETRINO o
dall'ELABORAZIONE.

La domanda non e' retorica: i due casi richiedono interventi opposti.

  A) VETRINO SOTTO-COLORATO
     L'originale e' gia' pallido. La pipeline lo preserva fedelmente,
     perche' allinea la TINTA (matrice di stain) e lascia intatte le
     CONCENTRAZIONI. Intervento: aggiungere una riscalatura delle
     concentrazioni verso un target (livello L2).

  B) ELABORAZIONE CHE SBIANCA
     L'originale ha colore, il processato no. Cause possibili:
       - stima della matrice fallita per quella WSI (cos_E basso):
         la deconvoluzione alloca male e l'eosina si perde
       - I0 sovrastimato: tutte le OD risultano piu' basse del vero
       - clipping np.maximum(C, 0) che scarta densita' reale
     Intervento: correggere la stima, NON riscalare a valle.

COSA MISURA
------------
Per gli stessi tile, prima e dopo:
  - densita' ottica media del tessuto (quanto e' "denso" il colore)
  - concentrazione di ematossilina ed eosina (99mo percentile)
  - frazione di pixel con eosina clippata a zero

Riferimenti pratici su H&E ben colorato:
  OD media del tessuto      0.4 - 1.2
  p99 delle concentrazioni  1.0 - 2.5  per ENTRAMBI gli stain
  clipping dell'eosina      sotto il 10%

Legge anche report_normalizzazione.json, se presente, per incrociare
il risultato con la qualita' della stima di quella WSI.

Uso
----
    python confronto_prima_dopo.py
====================================================================
"""

import os
import json

import numpy as np
import cv2

try:
    import config as C
    DATASET_DIR = C.DATASET_DIR
    CARTELLA_NORM = C.CARTELLA_IMG
except Exception:                                        # noqa: BLE001
    DATASET_DIR = r"E:\Tirocinio\Dataset\Dataset_Tiles"
    CARTELLA_NORM = "images_preprocessed"

CARTELLA_ORIG = "images"
REPORT_NORM = "report_normalizzazione.json"
N_TILE_PER_WSI = 8
SOGLIA_TESSUTO = 230          # sopra questa luminosita' = vetro
NOME_REPORT = "confronto_prima_dopo.json"

HE_CANONICA = np.array([[0.5626, 0.2159],
                        [0.7201, 0.8012],
                        [0.4062, 0.5581]])


def misura(cartella, files):
    """OD e concentrazioni H/E sui soli pixel di tessuto."""
    od_medie, c_h, c_e, clip_e, frazione_tessuto = [], [], [], [], []
    for f in files:
        img = cv2.imread(os.path.join(cartella, f))
        if img is None:
            continue
        px = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float64)
        I0 = np.maximum(np.percentile(px, 99, axis=0), 1.0)
        tessuto = px.mean(1) < SOGLIA_TESSUTO
        frazione_tessuto.append(float(tessuto.mean()))
        if tessuto.sum() < 100:
            continue
        OD = -np.log((px[tessuto] + 1.0) / I0)
        od_medie.append(float(OD.mean()))
        Craw = np.linalg.lstsq(HE_CANONICA, OD.T, rcond=None)[0]
        clip_e.append(float((Craw[1] <= 0).mean()))
        Cc = np.maximum(Craw, 0.0)
        c_h.append(float(np.percentile(Cc[0], 99)))
        c_e.append(float(np.percentile(Cc[1], 99)))
    if not od_medie:
        return None
    return {"od": float(np.mean(od_medie)),
            "H_p99": float(np.mean(c_h)),
            "E_p99": float(np.mean(c_e)),
            "clip_eosina": float(np.mean(clip_e)),
            "frazione_tessuto": float(np.mean(frazione_tessuto))}


def main():
    if not os.path.isdir(DATASET_DIR):
        print(f"DATASET_DIR non trovata: {DATASET_DIR}")
        return

    stime = {}
    path_rep = os.path.join(DATASET_DIR, REPORT_NORM)
    if os.path.exists(path_rep):
        try:
            rep = json.load(open(path_rep, encoding="utf-8"))
            stime = rep.get("wsi", rep) if isinstance(rep, dict) else {}
        except Exception:                                # noqa: BLE001
            pass

    wsi_folders = sorted(f for f in os.listdir(DATASET_DIR)
                         if os.path.isdir(os.path.join(DATASET_DIR, f))
                         and not f.startswith("_"))

    print("=" * 96)
    print("  CONFRONTO PRIMA / DOPO LA NORMALIZZAZIONE")
    print("=" * 96)
    print(f"  {'WSI':<12}| {'OD orig':>8}{'OD norm':>9} | {'H orig':>7}{'H norm':>8}"
          f" | {'E orig':>7}{'E norm':>8} | {'clipE':>7} | {'cos_E':>7}")
    print("  " + "-" * 92)

    righe = {}
    for wsi in wsi_folders:
        d_o = os.path.join(DATASET_DIR, wsi, CARTELLA_ORIG)
        d_n = os.path.join(DATASET_DIR, wsi, CARTELLA_NORM)
        if not (os.path.isdir(d_o) and os.path.isdir(d_n)):
            continue
        files = sorted(f for f in os.listdir(d_n) if f.endswith(".png"))
        files = [f for f in files if os.path.exists(os.path.join(d_o, f))]
        if not files:
            continue
        idx = np.linspace(0, len(files) - 1,
                          min(N_TILE_PER_WSI, len(files)), dtype=int)
        scelti = [files[i] for i in idx]

        mo, mn = misura(d_o, scelti), misura(d_n, scelti)
        if not (mo and mn):
            continue

        cos_e = None
        v = stime.get(wsi, {})
        if isinstance(v, dict):
            cos_e = v.get("cos_E_stimato", v.get("similarita_coseno_E"))

        righe[wsi] = {"orig": mo, "norm": mn, "cos_E": cos_e}
        s_cos = f"{cos_e:.3f}" if isinstance(cos_e, (int, float)) else "n/d"
        print(f"  {wsi:<12}| {mo['od']:>8.3f}{mn['od']:>9.3f} | "
              f"{mo['H_p99']:>7.2f}{mn['H_p99']:>8.2f} | "
              f"{mo['E_p99']:>7.2f}{mn['E_p99']:>8.2f} | "
              f"{100*mn['clip_eosina']:>6.1f}% | {s_cos:>7}")

    if not righe:
        print("\n  Nessuna WSI con entrambe le cartelle. Verificare i percorsi.")
        return

    od_o = np.array([r["orig"]["od"] for r in righe.values()])
    od_n = np.array([r["norm"]["od"] for r in righe.values()])
    e_o = np.array([r["orig"]["E_p99"] for r in righe.values()])
    e_n = np.array([r["norm"]["E_p99"] for r in righe.values()])
    h_n = np.array([r["norm"]["H_p99"] for r in righe.values()])
    clip = np.array([r["norm"]["clip_eosina"] for r in righe.values()])

    print()
    print("=" * 96)
    print("  VERDETTO")
    print("=" * 96)
    perdita_od = 100 * (1 - od_n.mean() / max(od_o.mean(), 1e-9))
    perdita_e = 100 * (1 - e_n.mean() / max(e_o.mean(), 1e-9))
    def _v(p):
        return f"perde il {p:.1f}%" if p > 0 else f"guadagna il {-p:.1f}%"
    print(f"  OD media del tessuto : originali {od_o.mean():.3f}  ->  "
          f"normalizzati {od_n.mean():.3f}   ({_v(perdita_od)})")
    print(f"  Eosina p99           : originali {e_o.mean():.2f}   ->  "
          f"normalizzati {e_n.mean():.2f}    ({_v(perdita_e)})")
    print(f"  Ematossilina p99     : normalizzati {h_n.mean():.2f}")
    print(f"  Clipping eosina      : {100*clip.mean():.1f}% dei pixel di tessuto")
    print()

    originale_pallido = e_o.mean() < 0.5
    processo_sbianca = perdita_e > 25

    if processo_sbianca:
        print("  IPOTESI B — L'ELABORAZIONE SBIANCA.")
        print(f"  L'eosina perde il {perdita_e:.0f}% passando per la pipeline.")
        print("  Non e' il vetrino: il colore c'era e si e' perso.")
        print()
        print("  Da controllare, in ordine:")
        print("   1. cos_E delle WSI peggiori (colonna a destra). Sotto 0.90 la")
        print("      stima del vettore eosina e' inaffidabile e la deconvoluzione")
        print("      alloca male: l'eosina finisce sull'ematossilina o nel nulla.")
        print("   2. HE_target: e' la MEDIANA elemento-per-elemento delle matrici")
        print("      stimate. Se le stime sono disperse, la mediana puo' avere le")
        print("      due colonne piu' simili fra loro di quanto lo siano nelle")
        print("      singole matrici, comprimendo la separazione fra gli stain.")
        print("      Verificare l'angolo fra le due colonne di HE_target: sotto i")
        print("      ~25 gradi la separazione e' degenere.")
        print("   3. Sostituire la mediana con la matrice H&E CANONICA come")
        print("      target. Si perde adattivita' ma si guadagna una separazione")
        print("      garantita e citabile (Ruifrok & Johnston 2001).")
    elif originale_pallido:
        print("  IPOTESI A — VETRINI SOTTO-COLORATI.")
        print(f"  Gia' gli originali hanno eosina p99 = {e_o.mean():.2f}, molto sotto")
        print("  l'intervallo tipico 1.0-2.5. La pipeline non peggiora nulla: per")
        print("  costruzione allinea la tinta e lascia intatte le concentrazioni.")
        print()
        print("  Intervento possibile: riscalare le CONCENTRAZIONI verso un")
        print("  target, cioe' il livello L2. Con una avvertenza importante:")
        print("   - amplifica anche il rumore, quindi il fattore va limitato")
        print("     (clip a 2-3x) e il target va preso dalla mediana del dataset,")
        print("     non da costanti esterne")
        print("   - non aggiunge informazione: i tile sono gia' uint8, riscalare")
        print("     distribuisce meglio i livelli esistenti, non ne crea di nuovi")
        print("   - il beneficio atteso e' sul CONDIZIONAMENTO (meno clipping,")
        print("     migliore uso del range dinamico), non sulla 'visibilita''")
        print()
        print("  Va introdotto come ABLATION e misurato, non dato per buono.")
    else:
        print("  NESSUNA ANOMALIA MARCATA.")
        print("  L'eosina e' preservata dalla pipeline e i valori originali sono")
        print("  nell'intervallo atteso. Lo sbiadimento percepito su singole tile")
        print("  riguarda probabilmente zone di tessuto genuinamente rado")
        print("  (stroma lasso, edema) e non richiede interventi.")

    peggiori = sorted(righe.items(), key=lambda kv: kv[1]["norm"]["E_p99"])[:6]
    print()
    print("  WSI con eosina piu' debole DOPO la normalizzazione:")
    for w, r in peggiori:
        s = f"{r['cos_E']:.3f}" if isinstance(r["cos_E"], (int, float)) else "n/d"
        print(f"    {w:<12} E_p99 {r['norm']['E_p99']:.2f}  "
              f"(orig {r['orig']['E_p99']:.2f})  cos_E {s}")

    try:
        with open(NOME_REPORT, "w", encoding="utf-8") as f:
            json.dump(righe, f, indent=2, ensure_ascii=False)
        print(f"\n  Report salvato in: {NOME_REPORT}")
    except OSError as e:
        print(f"\n  Impossibile salvare il report: {e}")


if __name__ == "__main__":
    main()