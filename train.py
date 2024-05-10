import os
import lightning.pytorch as pl

from setup import config
from utils.util_model import LightningTripletNet
from utils.util_dataset import LightningDataModule


def main():

    dataset = LightningDataModule(config)
    triplet_net = LightningTripletNet(config)
    trainer = pl.Trainer(accelerator="gpu",
                        devices=config.gpu_ids,
                        strategy="ddp",
                        max_epochs=config.total_epoch,
                        use_distributed_sampler=True,
                        precision="16-mixed",
                        logger=True,
                        profiler="simple",
                        default_root_dir=config.base_dir)
    trainer.fit(triplet_net, dataset)

if __name__ == '__main__':
    main()
