import os
import lightning.pytorch as pl
from setup import config
from utils.util_model import LightningTripletNet
from utils.util_dataset import LightningDataModule

def main():
    dataset = LightningDataModule(config)
    dataset.setup('test')
    triplet_net = LightningTripletNet.load_from_checkpoint("/home/student4/ai-place-recognition/experiments/sample_hardnordland/lightning_logs/lightning_logs/version_0/checkpoints/epoch=07_.ckpt", config=config, strict=False)
    
    trainer = pl.Trainer(accelerator="gpu",
                         devices=config.gpu_ids,
                         max_epochs=1,
                         strategy="ddp",
                         profiler="simple",
                         log_every_n_steps=1,
                         default_root_dir=config.base_dir)

    trainer.test(triplet_net, datamodule=dataset)

if __name__ == '__main__':
    main()
