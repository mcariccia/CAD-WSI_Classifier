"""
hed_augment.py
====================================================================
Augmentation per lo spazio H&E che ho scritto per gestire l'enorme 
divario di dominio tra il nostro dataset privato (molto pallido) e il
dataset pubblico MTCHI (molto intenso).

PERCHE' HO SCELTO QUESTO APPROCCIO
---------------------------------
Ho misurato l'intensita' dei soli pixel di tessuto (con bianco fisso I0 = 255):

                       OD mediana    H p95     E p95
    MTCHI                   0.580      8.90      0.78
    Mio privato (chiara)    0.166      0.88      0.12
    Mio privato (media)     0.200      ~1.0      ~0.25
    Rapporto MTCHI/privato    3.5x     10.1x      6.3x

Sicuramente c'e' una differenza biologica di base, ma un divario di
un ordine di grandezza sull'ematossilina non e' reale. Se addestro
il modello sul nostro dataset non funzionera' mai su MTCHI (e viceversa)
senza un'augmentation che copra questo sbalzo.

MOLTIPLICATIVO, NON ADDITIVO
-----------------------------
Ho preferito usare un fattore moltiplicativo ampio e un bias additivo 
quasi nullo per non distruggere l'ipercromasia nucleare:

    C' = C * g     Mantiene le proporzioni: un nucleo atipico resta 
                   piu' scuro dello stroma circostante.

    C' = C + b     Aggiunge un fondo finto: schiaccia la differenza tra 
                   nucleo e citoplasma. Soprattutto, dove C = 0 (e nel 
                   mio dataset succede spesso per l'eosina) crea colore 
                   dal nulla.

PERCHE' NON HO USATO skimage.color.rgb2hed
----------------------------------
La funzione base rgb2hed separa in TRE canali (H, E, DAB). Visto che il
DAB nei nostri vetrini non c'e', quel canale finisce per raccogliere il 
rumore e il residuo. Scalando solo H ed E e lasciando il DAB fermo, 
l'immagine non si scuriva quanto volevo.

Quindi ho implementato la separazione direttamente in densita' ottica
usando la matrice di Ruifrok & Johnston (2001), isolando il residuo:

    OD      = -log((px + 1) / I0)
    C       = max(pinv(HE) @ OD, 0)      concentrazioni, non negative
    residuo = OD - HE @ C                cio' che i due stain non spiegano
    OD'     = HE @ (C * alpha + beta) + residuo
    px'     = I0 * exp(-OD') - 1

Calcolo il residuo DOPO aver bloccato le concentrazioni negative a zero.
Questo mi garantisce due cose fondamentali:
  - Se imposto il fattore neutro (alpha=1, beta=0), riottengo la mia 
    immagine identica al pixel. Se non fosse cosi', avrei un bug.
  - Il rumore e gli artefatti (il "residuo") non vengono ingigantiti 
    quando alzo l'intensita'.
====================================================================
"""

import numpy as np
import albumentations as A

# Matrice H&E canonica. Colonne = [Ematossilina, Eosina], righe = [R,G,B],
# in densita' ottica con logaritmo naturale. Colonne a norma unitaria.
HE_CANONICA = np.array([[0.5626, 0.2159],
                        [0.7201, 0.8012],
                        [0.4062, 0.5581]], dtype=np.float32)
PINV_HE = np.linalg.pinv(HE_CANONICA).astype(np.float32)   # (2,3), precalcolata
I0 = np.float32(255.0)


class HEDIntensita(A.ImageOnlyTransform):
    """
    k_intensita : Il fattore GLOBALE di colorazione che ho impostato.
        Viene campionato in scala log-uniforme tra [1/k, k].
          1.4  -> Simula vetrini diversi del nostro stesso laboratorio
          2.0  -> Simula laboratori diversi
          3.0  -> Serve a coprire il divario col dataset MTCHI

        Ho scelto la scala log-uniforme perche' voglio che raddoppiare (*2) 
        e dimezzare (/2) abbiano la stessa probabilita'.
        Attenzione: valori alti (es. 3.0) costringono la rete a faticare di 
        piu' per trovare le feature giuste. Va testato con un'Ablation.

    k_rapporto : Lo sbilanciamento tra H ed E.
        Meglio tenerlo basso (1.2-1.4) perche' nei laboratori veri il 
        rapporto tra i coloranti varia molto meno dell'intensita' totale.

    bias_relativo : Bias additivo percentuale. L'ho messo come frazione, non in 
        assoluto. Visto che H ed E hanno magnitudini diversissime nel mio dataset, 
        un bias assoluto fisso sbilancerebbe troppo l'eosina. 

    scala_h, scala_e : Concentrazione tipica (99mo percentile) che ho 
        misurato sul mio dataset per i due coloranti, serve per il bias.
    """

    def __init__(self, k_intensita=2.0, k_rapporto=1.3, k_attenuazione=None,
                 bias_relativo=0.02, scala_h=1.02, scala_e=0.25, p=0.8):
        super().__init__(p=p)
        k_att = k_intensita if k_attenuazione is None else k_attenuazione
        self.log_k_int = float(np.log(max(k_intensita, 1.0 + 1e-9)))
        self.log_k_att = float(np.log(max(k_att, 1.0 + 1e-9)))
        self.log_k_rap = float(np.log(max(k_rapporto, 1.0 + 1e-9)))
        self.ampiezza_bias = np.array([bias_relativo * scala_h,
                                       bias_relativo * scala_e],
                                      dtype=np.float32)

    # ----------------------------------------------------------------
    @staticmethod
    def _scomponi(img):
        """RGB uint8 -> (concentrazioni (2,N), residuo (3,N), forma)."""
        h, w, _ = img.shape
        px = img.reshape(-1, 3).astype(np.float32)
        OD = -np.log((px + 1.0) / I0).T                  # (3, N)
        # Faccio il clipping QUI, prima del residuo: cosi' il residuo
        # assorbe la parte scartata e la ricostruzione a fattore neutro
        # e' esatta.
        C = np.maximum(PINV_HE @ OD, 0.0)                 # (2, N)
        residuo = OD - HE_CANONICA @ C                    # (3, N)
        return C, residuo, (h, w)

    @staticmethod
    def _ricomponi(C, residuo, forma):
        OD = HE_CANONICA @ C + residuo
        px = I0 * np.exp(-OD) - 1.0
        return np.clip(px.T, 0, 255).astype(np.uint8).reshape(*forma, 3)

    # ----------------------------------------------------------------
    def apply(self, img, **params):
        C, residuo, forma = self._scomponi(img)

        g = np.exp(np.random.uniform(-self.log_k_att, self.log_k_int))
        r = np.exp(np.random.uniform(-self.log_k_rap, self.log_k_rap))
        alpha = np.array([[g * r], [g / r]], dtype=np.float32)
        beta = np.random.uniform(-self.ampiezza_bias,
                                 self.ampiezza_bias).reshape(2, 1)

        # Le concentrazioni non possono essere negative: sotto zero non
        # corrispondono a nessuna quantita' fisica di colorante.
        C = np.maximum(C * alpha + beta, 0.0)
        return self._ricomponi(C, residuo, forma)

    def get_transform_init_args_names(self):
        return ("log_k_int", "log_k_att", "log_k_rap", "ampiezza_bias")
