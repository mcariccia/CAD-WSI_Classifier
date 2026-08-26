import numpy as np
import torch
from tqdm import tqdm

import config as C
import dati as D
import modello as M
import metriche as MT
from perdite import LossGerarchica


class FoldTrainer:

    def __init__(self, k, campioni, comp, idx_tr, idx_va, device,
                 classi_escluse=()):
        self.k = k
        self.device = device
        self.amp = (device.type == "cuda")
        self.ckpt = C.path_modello(k)
        self.classi_escluse = tuple(classi_escluse)

        (self.train_loader, self.val_loader,
         freq, self.wsi_va) = D.crea_loader(campioni, idx_tr, idx_va, comp)

        self.model = M.build_model(device, verbose=(k == 0))
        self.criterion = LossGerarchica(device, D.pesi_loss(freq))
        self.optimizer = M.crea_ottimizzatore(self.model)
        self.scheduler = M.crea_scheduler(self.optimizer)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.stage = 0

        self.history = {"train_loss": [], "val_loss": [], "train_miou": [],
                        "val_miou_fine": [], "val_miou_last": [],
                        "val_accbil": [], "val_sens_hsil": [],
                        "val_spec_hsil": []}

    # ----------------------------------------------------------------
    @torch.no_grad()
    def _valuta(self):
        self.model.eval()
        perdita = 0.0
        cm = MT.nuova_cm(device=self.device)
        cm_tile = MT.nuova_cm(device=self.device)
        cm_wsi = {}

        for images, masks, wsis in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self.amp):
                out = self.model(images)
                loss = self.criterion(out, masks)
            perdita += loss.item()
            pred = out.argmax(1)
            MT.aggiorna_cm(cm, pred, masks)
            MT.aggiorna_cm_tile(cm_tile, pred, masks)
            for w in dict.fromkeys(wsis):        
                sel = [j for j, x in enumerate(wsis) if x == w]
                cm_wsi.setdefault(w, MT.nuova_cm(device=self.device))
                MT.aggiorna_cm(cm_wsi[w], pred[sel], masks[sel])

        return (perdita / max(len(self.val_loader), 1),
                cm.cpu(), cm_tile.cpu(),
                {w: c.cpu() for w, c in cm_wsi.items()})

    # ----------------------------------------------------------------
    def _metrica_selezione(self, cm_wsi):
        """
        Media per WSI della mIoU LAST sulle sole classi presenti.
        GLASS escluso: il vetro e' banale da segmentare e gonfierebbe
        la metrica senza dire nulla sulla capacita' diagnostica.
        """
        vals = []
        for c in cm_wsi.values():
            v = MT.macro_presenti(MT.cm_to_last(c), escludi=(C.IDX_GLASS_LAST,))
            if not np.isnan(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    # ----------------------------------------------------------------
    def _epoca_train(self, epoch):
        self.model.train()
        M.freeze_encoder_bn(self.model)         
        tot_loss = 0.0
        cm_tr = MT.nuova_cm(device=self.device)

        loop = tqdm(self.train_loader, desc=f"  fold {self.k} ep {epoch+1}",
                    leave=False, ncols=78)
        for images, masks, _ in loop:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            if C.USA_CUTMIX:
                images, masks = D.cutmix(images, masks)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.amp):
                out = self.model(images)
                loss = self.criterion(out, masks)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), C.GRAD_CLIP)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            tot_loss += loss.item()
            MT.aggiorna_cm(cm_tr, out.argmax(1), masks)
            loop.set_postfix(loss=f"{loss.item():.3f}")

        return tot_loss / max(len(self.train_loader), 1), cm_tr.cpu()

    # ----------------------------------------------------------------
    def train(self):
        print(f"\n{'=' * 78}")
        print(f"  FOLD {self.k}   train {len(self.train_loader.dataset):,} tile"
              f"   |   val {len(self.val_loader.dataset):,} tile "
              f"/ {len(self.wsi_va)} WSI")
        print(f"  WSI in validation: {', '.join(self.wsi_va)}")
        if self.classi_escluse:
            print(f"  Classi escluse dalla metrica di selezione: "
                  f"{', '.join(C.CLASS_NAMES[c] for c in self.classi_escluse)}")
        print("=" * 78)
        print(f"  {'Ep':>3} | {'TrLoss':>7} | {'VaLoss':>7} | {'mIoUfin':>7} | "
              f"{'mIoUlast':>8} | {'accBil':>7} | {'sensHSIL':>8} | {'gap':>6} |")
        print("  " + "-" * 84)

        best, best_epoch = -1.0, -1
        for epoch in range(C.EPOCHS):
            if epoch in C.STAGE_SCHEDULE:
                self.stage = C.STAGE_SCHEDULE[epoch]
                M.set_encoder_stage(self.model, self.stage,
                                    verbose=(self.k == 0))

            train_loss, cm_tr = self._epoca_train(epoch)
            self.scheduler.step()
            train_miou = MT.macro(MT.iou_per_classe(cm_tr),
                                  escludi=(C.CLASSE_SFONDO,))

            val_loss, cm_va, cm_tile, cm_wsi = self._valuta()
            cm_last = MT.cm_to_last(cm_va)
            miou_fine = MT.macro(MT.iou_per_classe(cm_va),
                                 escludi=(C.CLASSE_SFONDO,) + self.classi_escluse)
            miou_last = self._metrica_selezione(cm_wsi)
            accbil = MT.accuratezza_bilanciata(
                cm_last, escludi=(C.IDX_GLASS_LAST,))
            hsil = MT.metriche_hsil(cm_last)
            gap = train_miou - miou_fine

            record = (not np.isnan(miou_last)) and miou_last > best
            print(f"  {epoch+1:>3} | {train_loss:>7.4f} | {val_loss:>7.4f} | "
                  f"{miou_fine:>7.4f} | {miou_last:>8.4f} | {accbil:>7.4f} | "
                  f"{hsil['sensibilita']:>8.4f} | {gap:>6.3f} |"
                  + (" RECORD" if record else ""))

            for chiave, v in zip(self.history.keys(),
                                 [train_loss, val_loss, train_miou, miou_fine,
                                  miou_last, accbil, hsil["sensibilita"],
                                  hsil["specificita"]]):
                self.history[chiave].append(v)

            if record:
                best, best_epoch = miou_last, epoch + 1
                torch.save(self.model.state_dict(), self.ckpt)
                self.cm_best = cm_va.clone()
                self.cm_tile_best = cm_tile.clone()
                self.cm_wsi_best = {w: c.clone() for w, c in cm_wsi.items()}

            if best_epoch > 0 and epoch + 1 - best_epoch >= C.PATIENCE:
                print(f"  Nessun record da {C.PATIENCE} epoche: fold interrotto.")
                break

        if best_epoch < 0:
            print(f"  ATTENZIONE fold {self.k}: nessun checkpoint salvato.")
            return None
        return self._riepilogo(best, best_epoch)

    # ----------------------------------------------------------------
    def _riepilogo(self, best, best_epoch):
        cm_last = MT.cm_to_last(self.cm_best)
        hsil = MT.metriche_hsil(cm_last)
        lsil_hsil = MT.discriminazione_lsil_hsil(cm_last)

        print(f"\n  Fold {self.k}: miglior mIoU LAST per WSI = {best:.4f} "
              f"(epoca {best_epoch})")
        print()
        MT.stampa_confusione(self.cm_best, C.CLASS_NAMES,
                             "confusione classi fini (% per riga)")
        print()
        MT.stampa_per_classe(self.cm_best, C.CLASS_NAMES)
        print()
        MT.stampa_confusione(cm_last, C.NOMI_LAST,
                             "confusione LAST (% per riga)")
        print(f"\n   HSIL vs non-HSIL:  sensibilita {hsil['sensibilita']:.4f}   "
              f"specificita {hsil['specificita']:.4f}   Dice {hsil['dice']:.4f}")
        print(f"   accuratezza bilanciata LAST (GLASS escluso): "
              f"{MT.accuratezza_bilanciata(cm_last, escludi=(C.IDX_GLASS_LAST,)):.4f}")
        print(f"   accuratezza per tile: {MT.accuratezza(self.cm_tile_best):.4f}"
              f"   bilanciata: "
              f"{MT.accuratezza_bilanciata(self.cm_tile_best):.4f}")
        print()
        MT.stampa_errori_principali(self.cm_best, C.CLASS_NAMES, top=6,
                                    titolo="Errori principali (classi fini)")
        print()
        MT.stampa_errori_principali(cm_last, C.NOMI_LAST, top=4,
                                    titolo="Errori principali (livello LAST)")

        per_wsi = {}
        print("\n   Per WSI (mIoU LAST sulle classi presenti):")
        for w, c in sorted(self.cm_wsi_best.items()):
            v = MT.macro_presenti(MT.cm_to_last(c), escludi=(C.IDX_GLASS_LAST,))
            per_wsi[w] = float(v)
            print(f"     {w:<18}{v:.4f}")

        return {
            "fold": self.k,
            "wsi_val": self.wsi_va,
            "best_epoch": best_epoch,
            "miou_last_per_wsi": float(best),
            "miou_fine": float(MT.macro(MT.iou_per_classe(self.cm_best),
                                        escludi=(C.CLASSE_SFONDO,))),
            "accbil_last": float(MT.accuratezza_bilanciata(
                cm_last, escludi=(C.IDX_GLASS_LAST,))),
            "accbil_tile": float(MT.accuratezza_bilanciata(self.cm_tile_best)),
            "hsil": hsil,
            "lsil_vs_hsil": lsil_hsil,
            "iou_fine": [None if np.isnan(v) else float(v)
                         for v in MT.iou_per_classe(self.cm_best).tolist()],
            "recall_fine": MT.recall_per_classe(self.cm_best),
            "errori_principali": MT.errori_principali(self.cm_best,
                                                      C.CLASS_NAMES, top=10),
            "cm_fine": self.cm_best.tolist(),
            "cm_last": cm_last.tolist(),
            "cm_tile": self.cm_tile_best.tolist(),
            "per_wsi": per_wsi,
            "history": self.history,
            "checkpoint": self.ckpt,
        }