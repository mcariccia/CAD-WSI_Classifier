import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==========================================
# CONFIGURAZIONE
# ==========================================
INPUT_FILE = "profilo_wsi.json"
NUM_ITERAZIONI = 2000000  # 2 milioni di iterazioni
RATIOS = [0.70, 0.15, 0.15]
CLASSI_NOMI = ["Sfondo", "CIN1", "Endocervical_glands", "HSIL", "Normal_Mucosa", "Stroma"]

# Fissa i seed per la riproducibilità scientifica
random.seed(42)
np.random.seed(42)

def calcola_pixel_assoluti(lista_wsi, profili_completi):
    """Calcola la somma bruta dei pixel per ogni classe in base alla lista di WSI fornita."""
    conteggi = np.zeros(6, dtype=np.float64)
    for nome in lista_wsi:
        dati = profili_completi[nome]
        for k, v in dati.items():
            conteggi[int(k)] += v
    return conteggi

def monte_carlo_split():
    with open(INPUT_FILE, "r") as f:
        profili = json.load(f)
        
    nomi_wsi = list(profili.keys())
    n_total = len(nomi_wsi)

    # Pre-calcoliamo i pixel totali dell'intero dataset per ogni classe
    totale_globale_per_classe = calcola_pixel_assoluti(nomi_wsi, profili)
    
    # Sostituiamo eventuali zeri con 1 per evitare divisioni per zero (safe-check)
    totale_globale_per_classe = np.where(totale_globale_per_classe == 0, 1, totale_globale_per_classe)
    
    # Suddivisione fissa a livello di pazienti/WSI (28 Train, 6 Val, 6 Test)
    n_train = int(n_total * RATIOS[0])
    n_val = int(n_total * RATIOS[1])
    
    best_score = float('inf')
    best_split = None
    best_ratios = None
    
    print(f"Avvio simulazione Monte Carlo su {n_total} WSI per {NUM_ITERAZIONI:,} iterazioni...")
    
    for i in tqdm(range(NUM_ITERAZIONI), desc="Calcolo Combinazioni"):
        
        # 1. Mischio a caso
        random.shuffle(nomi_wsi)
        
        # 2. Taglio la lista
        train_wsi = nomi_wsi[:n_train]
        val_wsi = nomi_wsi[n_train:n_train+n_val]
        test_wsi = nomi_wsi[n_train+n_val:]
        
        # 3. Calcolo pixel assoluti per split
        pixel_train = calcola_pixel_assoluti(train_wsi, profili)
        pixel_val = calcola_pixel_assoluti(val_wsi, profili)
        pixel_test = calcola_pixel_assoluti(test_wsi, profili)
        
        # 4. Calcolo la % di ogni classe finita nei tre set (Target: 70 / 15 / 15)
        ratio_train = pixel_train / totale_globale_per_classe
        ratio_val = pixel_val / totale_globale_per_classe
        ratio_test = pixel_test / totale_globale_per_classe
        
        # 5. METRICA DI ERRORE (Mean Squared Error)
        errore_train = np.sum((ratio_train - RATIOS[0]) ** 2)
        errore_val = np.sum((ratio_val - RATIOS[1]) ** 2)
        errore_test = np.sum((ratio_test - RATIOS[2]) ** 2)
        
        # 6. PENALITÀ ESTREME
        penalita = 0
        # Vogliamo che almeno il 2% (0.02) dei pixel di ogni classe patologica finisca in Val e Test
        # L'indice [1:] serve a escludere lo Sfondo (classe 0) da questa severa restrizione
        if np.any(ratio_val[1:] < 0.02) or np.any(ratio_test[1:] < 0.02):
            penalita += 1000
            
        score = errore_train + errore_val + errore_test + penalita
        
        # 7. AGGIORNAMENTO DEL RECORD
        if score < best_score:
            best_score = score
            best_split = (train_wsi.copy(), val_wsi.copy(), test_wsi.copy())
            best_ratios = (ratio_train, ratio_val, ratio_test)
        
    # Estrazione dei vincitori a fine ciclo
    train, val, test = best_split
    rt, rv, rts = best_ratios
    
    print("\n" + "="*75)
    print("🏆 MIGLIOR SPLIT STRATIFICATO TROVATO!")
    print("="*75)
    print(f"Punteggio di Errore (Basso = Migliore): {best_score:.6f}")
    print(f"Train WSI: {len(train)} | Val WSI: {len(val)} | Test WSI: {len(test)}")
    print("-" * 75)
    print(f"{'CLASSE':<20} {'TRAIN (Target 70%)':<22} {'VAL (Target 15%)':<22} {'TEST (Target 15%)'}")
    
    for i in range(6):
        print(f"{CLASSI_NOMI[i]:<20} {rt[i]*100:>7.2f}%                 {rv[i]*100:>7.2f}%                 {rts[i]*100:>7.2f}%")
        
    pd.DataFrame(train, columns=["WSI"]).to_csv("train_split.csv", index=False)
    pd.DataFrame(val, columns=["WSI"]).to_csv("val_split.csv", index=False)
    pd.DataFrame(test, columns=["WSI"]).to_csv("test_split.csv", index=False)
    print("\n✅ File CSV di split (Train, Val, Test) salvati con successo!")

if __name__ == "__main__":
    monte_carlo_split()