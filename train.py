import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from icdar import ICDARDataModule
from model import Model


@hydra.main(config_path="conf", config_name="train")
def main(cfg: DictConfig):
    datamodule = ICDARDataModule(**cfg.data)
    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    model = Model(cfg.model, cfg.optimizer, cfg.lambda_angle)
    trainer = Trainer(**cfg.trainer)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
