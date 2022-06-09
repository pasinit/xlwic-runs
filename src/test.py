import omegaconf
import hydra

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pl_data_modules import BasePLDataModule
from pl_modules import BasePLModule
import os
import logging
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['TOKENIZERS_PARALLELISM']='true'
os.environ['REQUESTS_CA_BUNDLE']='/etc/ssl/certs/ca-certificates.crt'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def test(conf: omegaconf.DictConfig) -> None:
    assert 'checkpoint_path' in conf
    checkpoint_path = conf.checkpoint_path
    # reproducibility
    pl.seed_everything(conf.train.seed)

    # data module declaration
    pl_data_module = BasePLDataModule(conf)

    # main module declaration
    pl_module = BasePLModule(conf)

    # trainer
    trainer: Trainer = hydra.utils.instantiate(conf.train.pl_trainer)

    pl_module = pl_module.load_from_checkpoint(checkpoint_path)

    # module test
    accuracies = {}
    for lang in pl_data_module.test_sets.keys():
        loader = pl_data_module.test_dataloader(lang)
        print('=' * 50)
        logger.info(f'Language: {lang}')
        trainer.test(pl_module, dataloaders=loader)
        accuracy = pl_module.test_accuracy
        accuracies[lang] = accuracy 
    
    for l, a in sorted(accuracies.items(), key=lambda elem: elem[0]):
        print(f'{l}: {a:.3f}')
    print(f'avg_acc: {sum(accuracies.values())/len(accuracies):.3f}')
    




@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    test(conf)


if __name__ == "__main__":
    main()
