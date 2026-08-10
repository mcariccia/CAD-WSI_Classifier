import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==========================================
# CONFIGURAZIONE
# ==========================================
input_file = "profilo_wsi.json"
NUM_ITERAZIONI = 1000000  # 1 milione è un ottimo compromesso tra velocità e precisione
RATIOS = [0.70, 0.15, 0.15] 

# FONDAMENTALE: Fissa i seed per la riproducibilità scientifica!
# Se un domani ri-esegui lo script, otterrai lo stesso identico split perfetto.
random.seed(42)
np.random.seed(42)

def calcola_distribuzione(lista_wsi, profili_completi):
    totale_pixel = 0
    # Usiamo solo le chiavi da 0 a 5, in linea con il JSON appena creato
    conteggi = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
    
    for nome in lista_wsi:
        dati = profili_completi[nome]
        for k, v in dati.items():
            conteggi[int(k)] += v
            totale_pixel += v
            
    if totale_pixel == 0: return np.zeros(6)
    return np.array([conteggi[i] for i in range(6)]) / totale_pixel

def monte_carlo_split():
    # Carica i dati estratti dal primo script
    with open(input_file, "r") as f:
        profili = json.load(f)
        
    nomi_wsi = list(profili.keys())
    n_total = len(nomi_wsi)

    totale_generale_tutti_i_pixel = sum(sum(wsi_data.values()) for wsi_data in profili.values())
    
    best_score = float('inf')
    best_split = None
    best_distributions = None
    
    print(f"Avvio simulazione Monte Carlo su {n_total} WSI per {NUM_ITERAZIONI:,} iterazioni...")
    
    # tqdm crea una barra di caricamento per farti vedere la velocità e il tempo rimanente
    for i in tqdm(range(NUM_ITERAZIONI), desc="Calcolo Combinazioni"):
        
        # 1. Mischio a caso l'ordine dei pazienti
        random.shuffle(nomi_wsi)
        
        # 2. Taglio la lista
        n_train = int(n_total * RATIOS[0])
        n_val = int(n_total * RATIOS[1])
        
        train_wsi = nomi_wsi[:n_train]
        val_wsi = nomi_wsi[n_train:n_train+n_val]
        test_wsi = nomi_wsi[n_train+n_val:]
        
        dist_train = calcola_distribuzione(train_wsi, profili)
        dist_val = calcola_distribuzione(val_wsi, profili)
        dist_test = calcola_distribuzione(test_wsi, profili)
        
        # 3. METRICA DI ERRORE (Mean Squared Error)
        diff_train_val = np.sum((dist_train - dist_val) ** 2)
        diff_train_test = np.sum((dist_train - dist_test) ** 2)
        
        # 4. PENALITÀ
        penalita = 0
        if np.any(dist_val < 0.005) or np.any(dist_test < 0.005):
            penalita += 1000  # Modificato in +=
            
        # ---> AGGIUNGI QUI: Controllo del volume del Train set <---
        totale_pixel_train = sum(sum(profili[w].values()) for w in train_wsi)
        vol_train = totale_pixel_train / totale_generale_tutti_i_pixel
        
        # Tolleranza: il Train deve contenere tra il 65% e il 75% dei pixel totali
        if vol_train < 0.65 or vol_train > 0.75:
            penalita += 1000
            
        score = diff_train_val + diff_train_test + penalita
        
    # Estrazione dei vincitori
    train, val, test = best_split
    d_train, d_val, d_test = best_distributions
    
    print("\n" + "="*60)
    print("🏆 MIGLIOR SPLIT TROVATO!")
    print("="*60)
    print(f"Punteggio di Errore (MSE + Penalità): {best_score:.6f}")
    print(f"Train WSI: {len(train)} | Val WSI: {len(val)} | Test WSI: {len(test)}")
    print("-" * 60)
    print(f"{'CLASSE':<20} {'TRAIN %':<12} {'VAL %':<12} {'TEST %':<12}")
    
    classi_nomi = ["Sfondo", "CIN1", "Endocervical_glands", "HSIL", "Normal_Mucosa", "Stroma"]
    for i in range(6):
        print(f"{classi_nomi[i]:<20} {d_train[i]*100:>7.2f}%    {d_val[i]*100:>7.2f}%    {d_test[i]*100:>7.2f}%")
        
    pd.DataFrame(train, columns=["WSI"]).to_csv("train_split.csv", index=False)
    pd.DataFrame(val, columns=["WSI"]).to_csv("val_split.csv", index=False)
    pd.DataFrame(test, columns=["WSI"]).to_csv("test_split.csv", index=False)
    print("\n✅ File CSV di split (Train, Val, Test) salvati con successo!")

if __name__ == "__main__":
    monte_carlo_split()