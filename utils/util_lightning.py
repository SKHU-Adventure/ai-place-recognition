import os
import lightning.pytorch as pl

def get_callbacks(config):
    tqdm_cb = pl.callbacks.TQDMProgressBar()
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        filename="{epoch:02d}_",
        save_last=True
    )
    early_stopping_cb = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )
    lr_monitor_cb = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [tqdm_cb, ckpt_cb, early_stopping_cb, lr_monitor_cb]
    return callbacks

def get_logger(config):
    # config['COMMON']['base_dir']로 수정
    base_dir = config['COMMON']['base_dir']
    logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(base_dir, "lightning_logs"))
    return logger