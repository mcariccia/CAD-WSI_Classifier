import numpy as np
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ==========================================
# CONFIGURAZIONE PERCORSI
# ==========================================
percorso_dataset = "E:\Tirocinio\Dataset\Dataset_Tiles"
output_file = "profilo_wsi.json"

def profila_wsi():
    dataset_path = Path(percorso_dataset)
    profili = {} 

    # Trova tutte le sottocartelle (le tue WSI)
    wsi_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if not wsi_folders:
        print("Errore: Nessuna cartella trovata nel percorso specificato.")
        return

    print(f"Inizio profilazione di {len(wsi_folders)} WSI...")

    # Ciclo principale sulle WSI con barra di caricamento
    for wsi_dir in tqdm(wsi_folders, desc="Analisi WSI"):
        nome_wsi = wsi_dir.name
        mask_path = wsi_dir / "masks"
        
        # Se la cartella masks non esiste, salta alla prossima WSI
        if not mask_path.exists():
            continue
            
        # Inizializza i contatori SOLO per le 6 classi utili (0 = Vetro, 1-5 = Tessuto annotato)
        conteggi = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        # Trova tutte le immagini PNG delle maschere
        files = list(mask_path.glob("*.png"))
        
        if not files:
            continue
            
        # Ciclo interno: analizza ogni singolo tile (mask)
        for f in files:
            # Carica l'immagine in memoria come matrice di numeri
            mask = np.array(Image.open(f))
            
            # np.unique trova i colori presenti e conta i pixel per ciascuno
            unici, quantita = np.unique(mask, return_counts=True)
            
            for u, q in zip(unici, quantita):
                # IL FILTRO: Se l'ID del colore è nel nostro dizionario (0-5), lo somma.
                # Se è 255 (bianco/ignore) o un artefatto, lo scarta matematicamente.
                if int(u) in conteggi:
                    conteggi[int(u)] += int(q)
        
        # Assegna i conteggi aggregati finali al nome della WSI corrente
        profili[nome_wsi] = conteggi

    # ==========================================
    # SALVATAGGIO IN JSON
    # ==========================================
    with open(output_file, "w") as f:
        # indent=4 formatta il JSON in modo leggibile (stile albero)
        json.dump(profili, f, indent=4)
        
    print(f"\nProfilazione completata con successo! Dati salvati in '{output_file}'")

if __name__ == "__main__":
    profila_wsi()