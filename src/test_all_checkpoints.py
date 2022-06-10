from copy import copy
from pathlib import Path
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
    assert 'checkpoint_dir' in conf
    checkpoint_dir = conf.checkpoint_dir
    # reproducibility
    pl.seed_everything(conf.train.seed)

    
    # main module declaration
    pl_module = BasePLModule(conf)
    
    # trainer
    trainer: Trainer = hydra.utils.instantiate(conf.train.pl_trainer)
    all_evaluations = dict()
    languages = None
    for checkpoint_path in Path(checkpoint_dir).rglob('*.ckpt'):
        pl_module = pl_module.load_from_checkpoint(checkpoint_path)
        hparams = pl_module.hparams
        target_identification = hparams.data.target_identification
        subword_combiner = hparams.model.subword_combiner
        words_combiner= hparams.model.words_combiner

        
        # data module declaration
        hparams2 = copy(hparams)
        hparams2.data.data_dir = conf.data.data_dir
        pl_data_module = BasePLDataModule(hparams2)
        
        # module test
        lang_accuracies = {}
        languages = sorted(pl_data_module.test_sets.keys())
        for lang in sorted(pl_data_module.test_sets.keys()):
            loader = pl_data_module.test_dataloader(lang)
            print('=' * 50)
            logger.info(f'Language: {lang}')
            trainer.test(pl_module, dataloaders=loader)
            accuracy = pl_module.test_accuracy
            lang_accuracies[lang] = accuracy 
        all_evaluations[(target_identification, subword_combiner, words_combiner)] = lang_accuracies
        
        print()    
    print('\t\t\t\t'+'\t'.join(languages))
    for configuration, lang_accs in all_evaluations.items():
        str_conf = "\t".join(configuration)
        str_accs = '\t'.join([f'{lang_accs[lang]*100:.2f}' for lang in languages])
        print(str_conf + '\t' + str_accs)
    print()




@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    test(conf)


if __name__ == "__main__":
    main()
