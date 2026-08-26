"""
modello.py
====================================================================
Costruzione del modello, gestione degli stage dell'encoder, scheduler.

Tre scelte non ovvie, tutte derivate da fallimenti misurati sul run
MTCHI:
  - encoder leggero (la capacita' e' il carburante della memorizzazione)
  - encoder mai sbloccato del tutto
  - BatchNorm dell'encoder sempre in eval
====================================================================
"""

import math

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

import config as C


def build_model(device, verbose=True):
    """
    U-Net con encoder leggero pre-addestrato su ImageNet.

    PERCHE' NON ResNet34. U-Net/ResNet34 ha 24.455.769 parametri. Con 40
    WSI, di cui molte mono o bi-classe, il numero di campioni
    indipendenti e' dell'ordine delle decine: la strada piu' economica
    per minimizzare la loss e' memorizzare le WSI, non imparare la
    morfologia. Sul run MTCHI questo produceva train_loss 0,0011 e
    train_mIoU 0,9991 con validazione in divergenza.

    La catena di fallback protegge da encoder non presenti
    nell'installazione locale di segmentation_models_pytorch.
    """
    ultimo, model = None, None
    for nome_enc in [C.ENCODER] + C.ENCODER_FALLBACK:
        try:
            model = smp.Unet(encoder_name=nome_enc, encoder_weights="imagenet",
                             in_channels=3, classes=C.NUM_CLASSES,
                             decoder_channels=C.DECODER_CHANNELS,
                             activation=None)
            if nome_enc != C.ENCODER and verbose:
                print(f"   encoder '{C.ENCODER}' non disponibile, "
                      f"uso '{nome_enc}'")
            break
        except Exception as e:                          
            ultimo, model = e, None
    if model is None:
        raise RuntimeError(f"Nessun encoder utilizzabile. Ultimo errore: {ultimo}")

    conv = model.segmentation_head[0]
    model.segmentation_head[0] = nn.Sequential(nn.Dropout2d(p=C.DROPOUT_HEAD),
                                               conv)
    model = model.to(device)
    set_encoder_stage(model, 0, verbose=verbose)
    if verbose:
        tot = sum(p.numel() for p in model.parameters())
        print(f"   parametri totali: {tot:,}   "
              f"(U-Net/ResNet34 ne aveva 24.455.769)")
    return model


def set_encoder_stage(model, stage, verbose=True):
    """
    stage 0 -> encoder interamente congelato, warm-up del decoder
    stage 1 -> ultimo FRAZIONE_SBLOCCO dei PARAMETRI sbloccato

    L'indicizzazione per posizione (invece di model.encoder.layer3)
    rende la funzione indipendente dall'architettura: MobileNetV3 non ha
    .layer3.
    """
    params = list(model.encoder.parameters())
    for p in params:
        p.requires_grad = False

    if stage >= 1:
        budget = sum(p.numel() for p in params) * C.FRAZIONE_SBLOCCO
        accumulato = 0
        for p in reversed(params):              
            if accumulato + p.numel() > budget and accumulato > 0:
                break
            p.requires_grad = True
            accumulato += p.numel()

    if verbose:
        tot = sum(p.numel() for p in model.parameters())
        tra = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   [stage {stage}] addestrabili: {tra:,} / {tot:,} "
              f"({100*tra/tot:.1f}%)")


def freeze_encoder_bn(model):
    """
    BatchNorm dell'encoder SEMPRE in eval, a qualunque stage.

    requires_grad=False non congela le statistiche running: in
    model.train() continuano ad aggiornarsi. Con batch 12 su tile mono o
    bi-classe, ogni batch e' dominato da una composizione casuale, le
    statistiche derivano, e in eval si usano quelle running: mismatch
    train/test garantito e invisibile nelle curve.

    Con l'encoder congelato la scelta pulita e' tenere le statistiche di
    ImageNet. Va chiamata DOPO model.train() a ogni epoca.
    """
    for m in model.encoder.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def crea_ottimizzatore(model):
    """
    Learning rate differenziati: l'encoder e' pre-addestrato e va
    ritoccato appena, il decoder parte da zero. Weight decay alto sul
    decoder, che e' l'unica parte che impara davvero e quindi l'unica
    che puo' memorizzare.
    """
    return torch.optim.AdamW([
        {"params": list(model.encoder.parameters()),
         "lr": C.LR_ENCODER, "weight_decay": C.WD_ENCODER},
        {"params": [p for n, p in model.named_parameters()
                    if not n.startswith("encoder.")],
         "lr": C.LR_DECODER, "weight_decay": C.WD_DECODER},
    ])


def crea_scheduler(optimizer, epochs=None, warmup=None):
    """Warmup lineare seguito da coseno che arriva a zero a fine training."""
    epochs = C.EPOCHS if epochs is None else epochs
    warmup = C.WARMUP_EP if warmup is None else warmup

    def fattore(e):
        if e < warmup:
            return (e + 1) / max(warmup, 1)
        t = (e - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fattore)