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
def train(conf: omegaconf.DictConfig) -> None:
    if 'dryrun' in conf:
        dryrun = conf.dryrun
    else:dryrun = False
    # reproducibility
    pl.seed_everything(conf.train.seed)

    # data module declaration
    pl_data_module = BasePLDataModule(conf)

    # main module declaration
    pl_module = BasePLModule(conf)

    # callbacks declaration
    callbacks_store = []

    if conf.train.early_stopping_callback is not None:
        early_stopping_callback: EarlyStopping = hydra.utils.instantiate(conf.train.early_stopping_callback)
        callbacks_store.append(early_stopping_callback)

    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint_callback: ModelCheckpoint = hydra.utils.instantiate(conf.train.model_checkpoint_callback)
        callbacks_store.append(model_checkpoint_callback)
    experiment_name = 'xlwic-' + conf.train.model_name
    save_dir = '/'.join(os.getcwd().split('/')[:-2])

    wandb_logger = pl_loggers.WandbLogger(name=experiment_name, 
            save_dir = save_dir, project='xlwic-reruns', 
            mode='dryrun' if dryrun else None)


    # trainer
    trainer: Trainer = hydra.utils.instantiate(conf.train.pl_trainer, callbacks=callbacks_store, logger=wandb_logger)

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)
    paths = [x for x in os.listdir(conf.train.model_checkpoint_callback.dirpath) if not x.startswith('.')]
    checkpoint_path = sorted(paths, key=lambda p : int(p.split('=')[1].split('-')[0]))[-1]
    checkpoint_path= os.path.join(conf.train.model_checkpoint_callback.dirpath, checkpoint_path)
    logger.info(f'Best checkpoint loaded from {checkpoint_path}')
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
    train(conf)


if __name__ == "__main__":
    main()
