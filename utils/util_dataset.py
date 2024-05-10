from torch.utils.data import DataLoader
import lightning.pytorch as pl

from datasets import get_dataset

class LightningDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.train_data_path = config.train_data_path
        self.test_data_path = config.test_data_path
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.data = config.data
        self.config = config

    def setup(self, stage: str):
        self.train_dataset = get_dataset(self.data, config=self.config, data_path=self.train_data_path)
        self.test_dataset = get_dataset(self.data, config=self.config, data_path=self.test_data_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)