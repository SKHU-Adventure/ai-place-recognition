import os
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from sklearn.metrics import roc_curve, auc

from setup import config, seed_worker
from utils.util_model import EmbedNet, QuadrupletNet
import utils.util_path as PATH
from utils.util_vis import draw_roc_curve
from utils.util_metric import AverageMeter
from datasets import get_dataset
from backbones import get_backbone
from models import get_model


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device_id = config.gpu_ids[rank]
    torch.cuda.set_device(device_id)

    train_dataset = get_dataset(config.data, config=config, data_path=config.train_data_path)
    test_dataset = get_dataset(config.data, config=config, data_path=config.test_data_path)

    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, config.batch_size, num_workers=config.num_workers, sampler=train_sampler, pin_memory=True, worker_init_fn=seed_worker)
    test_loader = torch.utils.data.DataLoader(test_dataset, config.batch_size, num_workers=config.num_workers, sampler=test_sampler, pin_memory=True, worker_init_fn=seed_worker)

    backbone = get_backbone(config.backbone)
    model = get_model(config.model)
    embed_net = EmbedNet(backbone, model)
    quadruplet_net = DDP(QuadrupletNet(embed_net).to(device_id), device_ids=[device_id])

    def quadruplet_loss(anc, pos, neg1, neg2, alpha1=0.5, alpha2=0.2):
        pos_dist = F.pairwise_distance(anc, pos, p=2)
        neg1_dist = F.pairwise_distance(anc, neg1, p=2)
        neg2_dist = F.pairwise_distance(neg1, neg2, p=2)
        loss1 = torch.relu(pos_dist - neg1_dist + alpha1)
        loss2 = torch.relu(pos_dist - neg2_dist + alpha2)
        loss = torch.mean(loss1 + loss2)
        return loss
    
    criterion = lambda a, p, n, an: quadruplet_loss(a, p, n, an, config.margin)
    optimizer = torch.optim.Adam(quadruplet_net.parameters(), lr=config.learning_rate)

    os.makedirs(PATH.CHECKPOINT, exist_ok=True)
    os.makedirs(PATH.VISUALIZATION, exist_ok=True)

    def train():
        quadruplet_net.train()
        losses = AverageMeter()

        for i, (anc, pos, neg, another_neg) in enumerate(train_loader):
            anc, pos, neg, another_neg = anc.to(device_id), pos.to(device_id), neg.to(device_id), another_neg.to(device_id)
            optimizer.zero_grad()
            anc_feat, pos_feat, neg_feat, another_neg_feat = quadruplet_net(anc, pos, neg, another_neg)
            loss = criterion(anc_feat, pos_feat, neg_feat, another_neg_feat)
            loss.backward()
            optimizer.step()

            losses.update(loss, anc.size(0))
        return losses.avg

    def validate():
        quadruplet_net.eval()
        losses = AverageMeter()
        dist_poses = AverageMeter()
        dist_neges1 = AverageMeter() 
        dist_neges2 = AverageMeter()  
        y_true = []
        y_scores = []

        with torch.no_grad():
            for i, (anc, pos, neg1, neg2) in enumerate(test_loader):
                anc, pos, neg1, neg2 = anc.to(device_id), pos.to(device_id), neg1.to(device_id), neg2.to(device_id)
                anc_feat, pos_feat, neg1_feat, neg2_feat = quadruplet_net(anc, pos, neg1, neg2)
                loss = criterion(anc_feat, pos_feat, neg1_feat, neg2_feat)
                dist_pos = F.pairwise_distance(anc_feat, pos_feat).cpu().numpy()
                dist_neg1 = F.pairwise_distance(anc_feat, neg1_feat).cpu().numpy()
                dist_neg2 = F.pairwise_distance(anc_feat, neg2_feat).cpu().numpy()

                losses.update(loss.item(), anc.size(0))
                dist_poses.update(np.mean(dist_pos), anc.size(0))
                dist_neges1.update(np.mean(dist_neg1), anc.size(0))
                dist_neges2.update(np.mean(dist_neg2), anc.size(0))
                y_true.extend([1] * anc.size(0) + [0] * 2 * anc.size(0))
                y_scores.extend(dist_pos)
                y_scores.extend(dist_neg1)
                y_scores.extend(dist_neg2)

        fpr, tpr, thresholds = roc_curve(y_true, -np.array(y_scores)) 
        roc_auc = auc(fpr, tpr)
        return losses.avg, dist_poses.avg, (dist_neges1.avg + dist_neges2.avg) / 2, roc_auc, fpr, tpr

    for epoch in range(1, config.total_epoch + 1):
        train_sampler.set_epoch(epoch)
        train_loss = train()
        avg_loss, avg_dist_pos, avg_dist_neg, roc_auc, fpr, tpr = validate()
        if rank == 0:
            print(f'[Epoch {epoch}] Train loss {train_loss:.4f}')
            print(f'[Epoch {epoch}] Validation loss {avg_loss:.4f}')
            print(f'[Epoch {epoch}] Average distance with positive sample: {avg_dist_pos:.4f}')
            print(f'[Epoch {epoch}] Average distance with negative sample: {avg_dist_neg:.4f}')
            print(f'[Epoch {epoch}] ROC AUC: {roc_auc:.4f}')
            draw_roc_curve(fpr, tpr, os.path.join(PATH.VISUALIZATION, f'roc_curve_e{epoch}.png'), roc_auc)
            torch.save(quadruplet_net.state_dict(), os.path.join(PATH.CHECKPOINT, f'{config.backbone}_{config.model}_{config.data}_checkpoint_e{epoch}.pth'))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
