import json

# Percorso del file generato in precedenza
input_file = "profilo_wsi.json"

def calcola_totali_globali():
    try:
        with open(input_file, "r") as f:
            profili = json.load(f)
    except FileNotFoundError:
        print(f"Errore: Il file '{input_file}' non esiste. Esegui prima Profila_WSI.py!")
        return

    # Inizializziamo i totali per le 6 classi (0-5)
    totale_generale_pixel = 0
    totali_per_classe = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    # Nomi leggibili per la stampa
    nomi_classi = {
        0: "Sfondo / Vetro",
        1: "CIN1",
        2: "Endocervical_glands",
        3: "HSIL",
        4: "Normal_Mucosa",
        5: "Stroma"
    }

    # Sommiamo i pixel di ogni WSI
    for nome_wsi, conteggi in profili.items():
        for classe_id, pixel in conteggi.items():
            # Convertiamo la chiave in int perché nel JSON è una stringa
            cid = int(classe_id)
            if cid in totali_per_classe:
                totali_per_classe[cid] += pixel
                totale_generale_pixel += pixel

    # --- STAMPA DEI RISULTATI ---
    print("="*50)
    print("DISTRIBUZIONE TOTALE DEL DATASET (40 WSI)")
    print("="*50)
    print(f"{'CLASSE':<25} {'PIXEL TOTALI':<20} {'PERCENTUALE':<10}")
    print("-"*50)

    for cid in range(6):
        pixel = totali_per_classe[cid]
        percentuale = (pixel / totale_generale_pixel * 100) if totale_generale_pixel > 0 else 0
        nome = nomi_classi[cid]
        
        # Formattiamo i numeri con le migliaia per leggerli meglio
        print(f"{nome:<25} {pixel:>20,} {percentuale:>10.2f}%")

    print("-"*50)
    print(f"{'TOTALE COMPLESSIVO:':<25} {totale_generale_pixel:>20,}")
    print("="*50)

if __name__ == "__main__":
    calcola_totali_globali()