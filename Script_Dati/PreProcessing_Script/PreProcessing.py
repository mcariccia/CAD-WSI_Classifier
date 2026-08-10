import cv2
import numpy as np
from sklearn.decomposition import NMF

class VahadaneNormalizer:
    def __init__(self, beta=0.15):
        """
        Inizializza i parametri per Vahadane.
        beta: Soglia di Densità Ottica (OD) per ignorare il background.
        """
        self.beta = beta
        self.HERef = None
        self.maxCRef = None
        # Inizializziamo il modello NMF (Non-Negative Matrix Factorization)
        self.nmf = NMF(n_components=2, init='nndsvd', random_state=42, max_iter=200)

    def _get_stain_vectors(self, I):
        I = I.astype(np.float64)
        
        # Convertiamo RGB in Optical Density (OD)
        OD = -np.log10((I + 1) / 256.0)
        
        # Rimuoviamo il background basandoci sulla soglia beta
        ODhat = OD[~np.any(OD < self.beta, axis=1)]
        
        if len(ODhat) == 0: 
            return None, None

        # Vahadane usa la NMF per estrarre le basi dei colori (Ematossilina ed Eosina)
        # H_matrix conterrà le concentrazioni, W_matrix i vettori colore
        H_matrix = self.nmf.fit_transform(ODhat)
        W_matrix = self.nmf.components_
        
        # Normalizziamo i vettori colore (W_matrix)
        W_matrix = W_matrix / np.linalg.norm(W_matrix, axis=1)[:, None]
        
        # Assicuriamoci che l'Ematossilina sia il primo vettore e l'Eosina il secondo
        # L'Ematossilina ha solitamente un valore più alto nel primo canale (Rosso in OD)
        if W_matrix[0, 0] < W_matrix[1, 0]:
            W_matrix = W_matrix[[1, 0], :]
            H_matrix = H_matrix[:, [1, 0]]
            
        HE = W_matrix.T
        
        # Calcoliamo le concentrazioni massime (99esimo percentile) per la normalizzazione
        Y = np.reshape(OD, (-1, 3)).T
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]
        maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
        
        return HE, maxC

    def fit(self, target_img):
        """Estrae e salva i vettori ideali da un'immagine di riferimento."""
        target_img = target_img.reshape((-1, 3))
        self.HERef, self.maxCRef = self._get_stain_vectors(target_img)

    def transform(self, img):
        """Applica i vettori di riferimento per normalizzare l'immagine in input."""
        if self.HERef is None or self.maxCRef is None:
            raise ValueError("Devi chiamare fit() prima di transform().")
            
        h, w, c = img.shape
        img_reshaped = img.reshape((-1, 3))
        
        # Otteniamo i vettori dell'immagine corrente
        HE, maxC = self._get_stain_vectors(img_reshaped)
        
        if HE is None: 
            return img
            
        Y = -np.log10((img_reshaped.astype(np.float64) + 1) / 256.0)
        
        # --- LOGICA IBRIDA (SCAN-like): Maschera HSV ---
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:, :, 1].reshape(-1)  
        v_channel = hsv[:, :, 2].reshape(-1)  
        is_background = (s_channel < 5) & (v_channel > 235)
        # -----------------------------------------------

        C = np.linalg.lstsq(HE, Y.T, rcond=None)[0]
        
        # Normalizziamo le concentrazioni rispetto ai massimi di riferimento
        maxC = np.where(maxC == 0, 1e-6, maxC) 
        C = C * (self.maxCRef / maxC)[:, np.newaxis]
        
        # Ricostruiamo l'immagine RGB
        Inorm = np.multiply(256, np.exp(-self.HERef.dot(C)))
        img_normalized = np.clip(Inorm.T, 0, 255).astype(np.uint8)
        
        # Manteniamo intatto il background originale
        img_normalized[is_background] = img_reshaped[is_background]
        
        return img_normalized.reshape((h, w, c))