from typing import Any, Union, List, Optional

from omegaconf import DictConfig
import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from xlwic_dataset import XLWIC_TESET_PATHS, XLWICDataset


class BasePLDataModule(pl.LightningDataModule):

    def __init__(self, conf: DictConfig):
        super().__init__()
        self.conf = conf
        self.xlwic_dir = conf.data.data_dir
        self.training_path = os.path.join(self.xlwic_dir, 'wic_english', 'train_en.txt')
        self.val_path = os.path.join(self.xlwic_dir, 'wic_english', 'valid_en.txt')
        self.test_paths = [os.path.join(self.xlwic_dir, p) for p in XLWIC_TESET_PATHS]
        self.tokenizer = AutoTokenizer.from_pretrained(conf.model.pretrained_model_name)
        self.training_set = XLWICDataset(self.training_path, tokenizer=self.tokenizer, split='train', language='EN')
        self.dev_set = XLWICDataset(self.val_path, tokenizer=self.tokenizer, split='dev', language='EN')
        self.test_sets = {}
        for path in self.test_paths:
            lang = path.split('_')[-1]
            dataset = XLWICDataset(os.path.join(path, f'{lang}_test_data.txt'), self.tokenizer, 
            answer_path=os.path.join(path, f'{lang}_test_gold.txt'), language=lang.upper(), split='test')
            self.test_sets[lang.upper()] = dataset

        
    def collate(self, examples):
        input_ids = [torch.LongTensor(e['input_ids']) for e in examples]
        labels = torch.Tensor([e['label'] for e in examples])
        indices_mask = [torch.LongTensor(e['indices_mask']) for e in examples]
        indices_mask = pad_sequence(indices_mask, batch_first=True, padding_value=0)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return {'input_ids': input_ids, 'labels': labels, 'indices_mask': indices_mask}

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.training_set, batch_size=self.conf.data.batch_size, shuffle=True, 
        collate_fn=self.collate, num_workers=self.conf.data.num_workers)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.dev_set, batch_size=self.conf.data.test_batch_size, shuffle=False, 
        collate_fn=self.collate, num_workers=self.conf.data.num_workers)

    def test_dataloader(self, lang, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        dataset = self.test_sets[lang.upper()]
        return DataLoader(dataset, batch_size=self.conf.data.test_batch_size, collate_fn=self.collate, num_workers=self.conf.data.num_workers, shuffle=False)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, *args, **kwargs) -> Any:
        aux = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device)
            aux[k] = v
        return aux
