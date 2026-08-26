"""
perdite.py
====================================================================
Loss gerarchica a quattro termini.

L'idea centrale: il modello continua a predire le CLASSI FINI, ma la
loss sa che quelle classi sono organizzate in una gerarchia clinica.
Un errore dentro il gruppo negativo costa poco; un errore che
attraversa il confine clinico costa molto.

Questo NON e' un collasso di classi. Le uscite restano NUM_CLASSES e i
termini a classi fini continuano a spingere Stroma lontano da Mucosa
nello spazio delle feature: la compattezza intra-classe fine e'
preservata. Cambia solo il COSTO relativo degli errori.

Rif. Bertinetto et al., "Making Better Mistakes" (CVPR 2020): usare
una gerarchia di etichette riduce la gravita' degli errori senza
degradare l'accuratezza fine.
====================================================================
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

import config as C


def logit_last(logits):
    """
    Marginalizza i logit delle classi fini sui gruppi LAST.

    logsumexp sui logit di un gruppo restituisce il logaritmo della
    probabilita' NON normalizzata del gruppo; applicando softmax al
    risultato si ottiene esattamente la marginale
        P(gruppo) = somma delle P(classi del gruppo)
    in modo numericamente stabile (verificato: errore max 3e-16 rispetto
    alla somma diretta delle probabilita').
    """
    return torch.stack([torch.logsumexp(logits[:, g], dim=1)
                        for g in C.GRUPPI_LAST], dim=1)


class LossGerarchica(nn.Module):
    """
    TERMINE 1 — CE sulle classi fini, pesata, con label smoothing.
        Il modello deve comunque distinguere le classi fini: e' la
        segmentazione diagnostica, e le Ghiandole endocervicali (95,6%
        di recall nel run precedente) sono un segnale di supervisione
        affidabile che non conviene buttare.

        label_smoothing: l'etichetta CIN1 e' intrinsecamente incerta.
        Nell'ALTS un'interpretazione bioptica di CIN1 fu confermata dal
        panel di controllo qualita' solo nel 42,6% dei casi (Stoler &
        Schiffman 2001, JAMA 285:1500-1505, kappa 0,46). Addestrare a
        probabilita' 1,0 su un'etichetta la cui riproducibilita' umana
        e' ~40% e' un errore statistico, non un dettaglio.

    TERMINE 2 — Dice sulle classi fini.
        Compensa lo sbilanciamento in modo complementare alla CE.

    TERMINE 3 — CE sulle probabilita' MARGINALIZZATE LAST.
        Ghiandole, Mucosa e Stroma finiscono nello stesso gruppo
        NEGATIVE, quindi confonderle non costa nulla in questo termine.
        Nel run a 6 classi l'errore Stroma -> Mucosa era il 60,4% e
        dominava la metrica, pur essendo clinicamente irrilevante:
        entrambe mappano su "negativo" al livello LAST.

        Effetto collaterale utile: dove la supervisione e' debole
        (Stroma e' annotato in ~10 WSI su 40, con l'83% dei pixel in 2
        sole WSI) il modello puo' ripiegare su "negativo" con costo
        ridotto invece di essere costretto a indovinare, riducendo il
        gradiente rumoroso che si propagherebbe nell'encoder condiviso.

    TERMINE 4 — Vincolo di tessuto sui pixel 255.
        Di quei pixel si sa una cosa certa: sono tessuto delimitato ma
        non classificato, quindi NON sono vetro. Termine
        -log(1 - P(Sfondo)), verificato equivalente a
        -(logsumexp(logit_tessuto) - logsumexp(logit_tutti)).

        Oggi quei pixel non producono alcun gradiente: sono una
        frazione enorme del dataset buttata via. Qui addestrano
        l'encoder a separare tessuto da vetro. E' la forma minima e
        sicura di supervisione parziale — versione debole della
        marginal loss di Shi et al. 2021 (Med Image Anal 70:101979),
        senza le sue ipotesi piu' rischiose sulla distribuzione delle
        classi non annotate.
    """

    def __init__(self, device, pesi=None):
        super().__init__()
        self.device = device
        w = pesi.to(device) if pesi is not None else None

        try:
            self.ce = nn.CrossEntropyLoss(weight=w,
                                          ignore_index=C.IGNORE_INDEX,
                                          label_smoothing=C.LABEL_SMOOTH)
        except TypeError:                       # torch < 1.10
            self.ce = nn.CrossEntropyLoss(weight=w, ignore_index=C.IGNORE_INDEX)

        self.ce_last = nn.CrossEntropyLoss(ignore_index=C.IGNORE_INDEX)
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True,
                                        ignore_index=C.IGNORE_INDEX)

        self.register_buffer("lut_last", C.LUT_LAST.to(device))
        self.register_buffer("idx_tessuto",
                             torch.tensor(C.CLASSI_TESSUTO, device=device))

    # ----------------------------------------------------------------
    def _target_last(self, y):
        """Rimappa le etichette fini sui gruppi LAST, preservando l'ignore."""
        valido = y != C.IGNORE_INDEX
        t = torch.where(valido, y, torch.zeros_like(y))
        t = self.lut_last[t]
        return torch.where(valido, t, torch.full_like(t, C.IGNORE_INDEX))

    def _vincolo_tessuto(self, logits, y):
        """-log P(non vetro) sui soli pixel marcati come tessuto non classificato."""
        m = y == C.IGNORE_INDEX
        if m.sum() == 0:
            return logits.sum() * 0.0
        log_tessuto = torch.logsumexp(logits[:, self.idx_tessuto], dim=1)
        log_tutto = torch.logsumexp(logits, dim=1)
        return -(log_tessuto - log_tutto)[m].mean()

    # ----------------------------------------------------------------
    def forward(self, logits, y):
        annotati = (y != C.IGNORE_INDEX).sum()

        if annotati == 0:
            perdita = logits.sum() * 0.0
        else:
            perdita = C.W_CE * self.ce(logits, y) + C.W_DICE * self.dice(logits, y)
            if C.W_LAST > 0:
                perdita = perdita + C.W_LAST * self.ce_last(
                    logit_last(logits), self._target_last(y))

        if C.W_TESSUTO > 0:
            perdita = perdita + C.W_TESSUTO * self._vincolo_tessuto(logits, y)

        return perdita

    # ----------------------------------------------------------------
    def dettaglio(self, logits, y):
        """Valore dei singoli termini, per diagnosticare quale domina."""
        with torch.no_grad():
            out = {}
            if (y != C.IGNORE_INDEX).sum() > 0:
                out["ce"] = float(self.ce(logits, y))
                out["dice"] = float(self.dice(logits, y))
                if C.W_LAST > 0:
                    out["last"] = float(self.ce_last(logit_last(logits),
                                                     self._target_last(y)))
            if C.W_TESSUTO > 0:
                out["tessuto"] = float(self._vincolo_tessuto(logits, y))
            return out