from torch.utils.data import DataLoader
import torch
from torchinfo import summary
from feeder import FeederINCLUDE
from aagcn import Model
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from augumentation import Rotate, Left, Right, GaussianNoise, Compose
from torch.utils.data import random_split

if __name__ == '__main__':

    # Hyper parameter tuning : batch_size, learning_rate, weight_decay
    config = {'batch_size': 2, 'learning_rate': 0.0137296, 'weight_decay': 0.000150403}
    # Load device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load model
    model = Model(num_class=101, num_point=46, num_person=1, in_channels=2,
                graph_args = {"layout" :"mediapipe_two_hand", "strategy": "spatial"},
                learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])

    # Callback PL
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            monitor="valid_accuracy",
            mode="max",
            every_n_epochs = 2,
            filename='{epoch}-{valid_accuracy:.2f}-wsl_100-aagcn-{fold}'
        ),
    ]
    # Augument 
    batch_size = config["batch_size"]
    transforms = Compose([
        Rotate(15, 80, 25, (0.5, 0.5))
    ])

    # Dataset class
    train_dataset = FeederINCLUDE(data_path=f"wsl100_train_data_preprocess.npy", label_path=f"wsl100_train_label_preprocess.npy",
                            transform=transforms)
    test_dataset = FeederINCLUDE(data_path=f"wsl100_test_data_preprocess.npy", label_path=f"wsl100_test_label_preprocess.npy")
    valid_dataset = FeederINCLUDE(data_path=f"wsl100_valid_data_preprocess.npy", label_path=f"wsl100_valid_label_preprocess.npy")

    # DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    val_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)

    specific_batch = next(iter(train_dataloader))
    print("Input shape ", specific_batch[0].shape)
    print("Data loader success")
    # Trainer PL
    trainer = pl.Trainer(max_epochs = 100, accelerator="auto", check_val_every_n_epoch = 1, 
                       devices = 1, callbacks=callbacks)
                    #  , logger=wandb_logger) # wandb
    trainer.fit(model, train_dataloader, val_dataloader)
    # Test PL (When test find the right ckpt_path and comment code line 58)
    # trainer.test(model, test_dataloader, ckpt_path="checkpoints/epoch=61-valid_accuracy=0.91-vsl_100-aagcn-2hand+preprocessing_keypoint+augment(v1).ckpt", 
                # verbose=True)