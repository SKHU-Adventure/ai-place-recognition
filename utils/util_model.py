import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torchvision import transforms
from torchvision.utils import save_image
from backbones import get_backbone
from models import get_model
import os

class EmbedNet(pl.LightningModule):
    def __init__(self, backbone, model):
        super(EmbedNet, self).__init__()
        self.backbone = backbone
        self.model = model

    def forward(self, x):
        x = self.backbone(x)
        embedded_x = self.model(x)
        return embedded_x

class TripletNet(pl.LightningModule):
    def __init__(self, embed_net):
        super(TripletNet, self).__init__()
        self.embed_net = embed_net

    def forward(self, a, p, n):
        embedded_a = self.embed_net(a)
        embedded_p = self.embed_net(p)
        embedded_n = self.embed_net(n)
        return embedded_a, embedded_p, embedded_n

    def feature_extract(self, x):
        return self.embed_net(x)

class LightningTripletNet(pl.LightningModule):
    def __init__(self, config):
        super(LightningTripletNet, self).__init__()
        self.config = config
        backbone = get_backbone(self.config.backbone)
        model = get_model(self.config.model)
        embed_net = EmbedNet(backbone, model)
        self.triplet_net = TripletNet(embed_net)
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, a, p, n):
        return self.triplet_net(a, p, n)

    def training_step(self, batch, batch_idx):
        a, p, n = batch
        embedded_a, embedded_p, embedded_n = self.triplet_net(a, p, n)
        loss = nn.TripletMarginLoss(margin=self.config.margin)(embedded_a, embedded_p, embedded_n)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        a, p, n = batch
        embedded_a, embedded_p, embedded_n = self.triplet_net(a, p, n)
        loss = nn.TripletMarginLoss(margin=self.config.margin, reduction='none')(embedded_a, embedded_p, embedded_n)
        dist_pos = F.pairwise_distance(embedded_a, embedded_p)
        dist_neg = F.pairwise_distance(embedded_a, embedded_n)
        self.validation_step_outputs.append((loss, dist_pos, dist_neg))
        return loss, dist_pos, dist_neg

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer

    def on_validation_epoch_end(self):
        loss = torch.cat([x for x, y, z in self.validation_step_outputs]).detach().cpu().numpy()
        dist_pos = torch.cat([y for x, y, z in self.validation_step_outputs]).detach().cpu().numpy()
        dist_neg = torch.cat([z for x, y, z in self.validation_step_outputs]).detach().cpu().numpy()
        avg_loss = np.mean(loss)
        avg_dist_pos = np.mean(dist_pos)
        avg_dist_neg = np.mean(dist_neg)
        self.validation_step_outputs.clear()
        self.log("val_loss", avg_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("dist_pos", avg_dist_pos, prog_bar=True, logger=True, sync_dist=True)
        self.log("dist_neg", avg_dist_neg, prog_bar=True, logger=True, sync_dist=True)
        return avg_loss, avg_dist_pos, avg_dist_neg