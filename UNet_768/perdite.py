"""
perdite.py
====================================================================
Loss gerarchica a cinque termini. Nessuna dipendenza da config: tutti i
parametri arrivano dal costruttore (config.parametri_loss()), cosi' il
modulo e' testabile da solo e i parametri effettivi di un run
coincidono per costruzione con quelli scritti nel report.

  SCE   Symmetric Cross Entropy (Wang et al. 2019). La RCE e' scritta
        nella forma chiusa A*(1 - p_vera) invece che con il one-hot
        clampato a 1e-4: con quel clamp la costante valeva log(1e-4) =
        9.21 e il termine dominava la CE di 5-9 volte, rendendo di
        fatto inerti i pesi di classe. Con A = 4 (valore di Wang) e i
        pesi applicati anche alla RCE il bilanciamento torna a
        funzionare. Una loss dominata da un termine tipo MAE e' robusta
        al rumore ma sotto-adatta le classi rare (Ghosh et al. 2017),
        che qui vuol dire CIN1.

  FTL   Focal Tversky, mediata SOLO sulle classi presenti nel batch.
        Per una classe assente si ha TP = FN = 0 e il contributo satura
        a 1 qualunque cosa faccia la rete: con il 51% di tile
        mono-classe meta' del termine era un plateau il cui unico
        gradiente diceva "togli massa dalle classi rare". CIN1,
        presente in 10 WSI su 34, era la piu' penalizzata.
        Convenzione: alpha pesa i FP, beta i FN (alpha=0.3, beta=0.7
        e' il valore standard in letteratura).

  LAST  Cross entropy su GLASS/NEGATIVE/LSIL/HSIL, ottenuta
        marginalizzando i logit fini sui gruppi via logsumexp.

  COSTO Rischio atteso sotto matrice dei costi (Elkan 2001). Sostituisce
        la penalita' binaria precedente, che assegnava costo ZERO a
        CIN1 -> HSIL e aveva quindi un minimo degenere: qualunque
        distribuzione sulle due classi lesionali la annullava, e la
        rete sceglieva la piu' frequente. Con Omega graduata il minimo
        e' unico e nella classe corretta.

  VETRO Supervisione binaria vetro/tessuto sui pixel non annotati, con
        la pseudo-etichetta prodotta da dati.marca_vetro. Sostituisce
        il vincolo precedente, che imponeva che TUTTI i pixel non
        annotati fossero tessuto: un prior falso ovunque ci sia vetro
        non annotato.

NOTA SULLA MEMORIA. La softmax sui soli pixel validi si calcola UNA
volta in forward e si passa ai tre termini che ne hanno bisogno. Con
batch 16 a 512 px quel tensore e' circa 100 MB: calcolarlo tre volte,
come nella versione precedente, triplicava il picco senza motivo.
====================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def logit_last(logits, gruppi_last):
    """Marginalizza i logit delle classi fini sui gruppi LAST."""
    return torch.stack([torch.logsumexp(logits[:, g], dim=1)
                        for g in gruppi_last], dim=1)


class SymmetricCrossEntropy(nn.Module):
    """
    alpha pesa la CE, beta la RCE. La RCE e' pesata per classe come la
    CE: senza, i pesi calcolati in dati.pesi_loss influenzerebbero solo
    una frazione minoritaria del gradiente.
    """
    def __init__(self, num_classes, alpha=1.0, beta=0.3, a=4.0, weight=None,
                 ignore_index=255, label_smoothing=0.0):
        super().__init__()
        self.alpha, self.beta, self.a = alpha, beta, a
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,
                                      label_smoothing=label_smoothing)
        self.register_buffer(
            "w", weight.clone() if weight is not None
            else torch.ones(num_classes))

    def forward(self, logits, targets, prob_v=None, t_v=None):
        """
        prob_v: [N, K] softmax sui soli pixel validi, gia' calcolata.
        t_v:    [N]    etichette degli stessi pixel.
        Se assenti vengono ricavate qui, cosi' la classe resta usabile
        da sola nei test.
        """
        ce_loss = self.ce(logits, targets)

        if prob_v is None or t_v is None:
            valido = targets != self.ignore_index
            if valido.sum() == 0:
                return self.alpha * ce_loss
            prob_v = F.softmax(logits, dim=1).permute(0, 2, 3, 1)[valido]
            t_v = targets[valido]
        if t_v.numel() == 0:
            return self.alpha * ce_loss

        p_vera = prob_v.gather(1, t_v[:, None]).squeeze(1).clamp(1e-7, 1.0)
        w = self.w.to(prob_v.dtype)[t_v]
        rce = (self.a * (1.0 - p_vera) * w).sum() / w.sum().clamp(min=1e-8)
        return self.alpha * ce_loss + self.beta * rce


class FocalTverskyLoss(nn.Module):
    """alpha pesa i Falsi Positivi, beta i Falsi Negativi."""
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.333, smooth=1.0):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.gamma, self.smooth = gamma, smooth

    def forward(self, prob_v, t_v, num_classes):
        if t_v.numel() == 0:
            return prob_v.sum() * 0.0

        t = F.one_hot(t_v, num_classes=num_classes).to(prob_v.dtype)

        tp = (prob_v * t).sum(dim=0)
        fp = (prob_v * (1 - t)).sum(dim=0)
        fn = ((1 - prob_v) * t).sum(dim=0)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth)
        ft = (1 - tversky) ** self.gamma

        # Una classe assente dal batch ha TP = FN = 0: il suo indice di
        # Tversky non misura nulla e satura. Si media sulle presenti.
        presenti = t.sum(dim=0) > 0
        if not bool(presenti.any()):
            return prob_v.sum() * 0.0
        return ft[presenti].mean()


class LossGerarchica(nn.Module):
    def __init__(self, device, pesi=None, *,
                 num_classes, ignore_index, ignore_vetro,
                 gruppi_last, lut_last, classi_tessuto, classi_vetro,
                 w_ce, w_dice, w_last, w_clinica, w_tessuto,
                 sce_alpha, sce_beta, sce_a, label_smooth,
                 ftl_alpha, ftl_beta, ftl_gamma, ftl_smooth,
                 matrice_costi=None):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ignore_vetro = ignore_vetro
        self.gruppi_last = [list(g) for g in gruppi_last]
        self.w_ce, self.w_dice = w_ce, w_dice
        self.w_last, self.w_clinica, self.w_tessuto = w_last, w_clinica, w_tessuto

        w = pesi.to(device) if pesi is not None else None
        self.sce = SymmetricCrossEntropy(
            num_classes, alpha=sce_alpha, beta=sce_beta, a=sce_a, weight=w,
            ignore_index=ignore_index, label_smoothing=label_smooth).to(device)
        self.ftl = FocalTverskyLoss(alpha=ftl_alpha, beta=ftl_beta,
                                    gamma=ftl_gamma, smooth=ftl_smooth)
        self.ce_last = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.register_buffer("lut_last",
                             torch.as_tensor(lut_last, dtype=torch.long,
                                             device=device))
        self.register_buffer("idx_tessuto",
                             torch.as_tensor(classi_tessuto, dtype=torch.long,
                                             device=device))
        self.register_buffer("idx_vetro",
                             torch.as_tensor(classi_vetro, dtype=torch.long,
                                             device=device))
        omega = (torch.as_tensor(matrice_costi, dtype=torch.float32)
                 if matrice_costi is not None
                 else torch.zeros(num_classes, num_classes))
        if tuple(omega.shape) != (num_classes, num_classes):
            raise ValueError(
                f"MATRICE_COSTI {tuple(omega.shape)}, attesa "
                f"({num_classes}, {num_classes}).")
        self.register_buffer("omega", omega.to(device))

    # ----------------------------------------------------------------
    def _solo_annotati(self, y):
        """254 e 255 sono entrambi 'non annotato' per i termini supervisionati."""
        return torch.where(y >= self.num_classes,
                           torch.full_like(y, self.ignore_index), y)

    def _target_last(self, y):
        valido = y < self.num_classes
        t = torch.where(valido, y, torch.zeros_like(y))
        t = self.lut_last[t]
        return torch.where(valido, t, torch.full_like(t, self.ignore_index))

    def _costo_atteso(self, prob_v, t_v):
        """Rischio atteso sotto la matrice dei costi."""
        if t_v.numel() == 0:
            return prob_v.sum() * 0.0
        return (prob_v * self.omega.to(prob_v.dtype)[t_v]).sum(dim=1).mean()

    def _vincolo_vetro(self, logits, y):
        """
        Supervisione binaria vetro/tessuto sui pixel non annotati.

        log P(tessuto) e log P(vetro) si calcolano come due logsumexp
        separati sui rispettivi gruppi di classi, non uno per differenza
        dall'altro: sotto autocast in fp16 exp(log_pt) arrotonda a 1.0
        appena la rete e' confidente, e log1p(-1.0) darebbe -inf con
        conseguente NaN. Il cast a float32 e' locale e costa nulla.
        """
        m = y >= self.num_classes
        if m.sum() == 0:
            return logits.sum() * 0.0
        lg = logits.float()
        log_a = torch.logsumexp(lg, dim=1)
        log_t = torch.logsumexp(lg[:, self.idx_tessuto], dim=1) - log_a
        log_v = torch.logsumexp(lg[:, self.idx_vetro], dim=1) - log_a
        t = (y[m] == self.ignore_index).float()          # 1 = tessuto
        return -(t * log_t[m] + (1.0 - t) * log_v[m]).mean()

    # ----------------------------------------------------------------
    def forward(self, logits, y):
        y_sup = self._solo_annotati(y)
        valido = y_sup != self.ignore_index
        perdita = logits.sum() * 0.0

        if bool(valido.any()):
            prob_v = F.softmax(logits, dim=1).permute(0, 2, 3, 1)[valido]
            t_v = y_sup[valido]

            perdita = perdita + self.w_ce * self.sce(logits, y_sup, prob_v, t_v)
            perdita = perdita + self.w_dice * self.ftl(prob_v, t_v,
                                                       self.num_classes)
            if self.w_last > 0:
                perdita = perdita + self.w_last * self.ce_last(
                    logit_last(logits, self.gruppi_last), self._target_last(y))
            if self.w_clinica > 0:
                perdita = perdita + self.w_clinica * self._costo_atteso(prob_v,
                                                                        t_v)

        if self.w_tessuto > 0:
            perdita = perdita + self.w_tessuto * self._vincolo_vetro(logits, y)

        return perdita

    @torch.no_grad()
    def dettaglio(self, logits, y):
        """Contributo dei singoli termini, non pesato. Per diagnosi."""
        y_sup = self._solo_annotati(y)
        valido = y_sup != self.ignore_index
        out = {}
        if bool(valido.any()):
            prob_v = F.softmax(logits, dim=1).permute(0, 2, 3, 1)[valido]
            t_v = y_sup[valido]
            out["sce"] = float(self.sce(logits, y_sup, prob_v, t_v))
            out["ftl"] = float(self.ftl(prob_v, t_v, self.num_classes))
            out["costo"] = float(self._costo_atteso(prob_v, t_v))
            if self.w_last > 0:
                out["last"] = float(self.ce_last(
                    logit_last(logits, self.gruppi_last), self._target_last(y)))
        if self.w_tessuto > 0:
            out["vetro"] = float(self._vincolo_vetro(logits, y))
        return out