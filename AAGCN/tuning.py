# Hyper parameter tuning : batch_size, learning_rate, weight_decay
from feeder import FeederINCLUDE
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import Model, Model_TwoStream
# from agcn import Model
# from aagcn import Model
# from decoupleGCN.decouple_gcn import Model
import ray
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
from train_vsl import train_dataset, test_dataset, valid_dataset
# from main import train_dataset, test_dataset, valid_dataset

class VSLDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8):
        super().__init__()
        self.batch_size = batch_size
    
    def setup(self, stage = None):
        self.train_data = train_dataset
        self.test_data = test_dataset
        self.valid_data = valid_dataset
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

# {'batch_size': 8, 'learning_rate': 0.00020819749402816755, 'weight_decay': 0.00017443377277796699}
def train_vsl(config):
    # model = Model(2, 10, graph_args = {"layout" :"mediapipe", "strategy": "spatial"}, 
    #               edge_importance_weighting=True, learning_rate=config["learning_rate"], 
    #               weight_decay=config["weight_decay"])
    
    model = Model(num_class=100, num_point=46, num_person=1, in_channels=2,
                  graph_args = {"layout" :"mediapipe_two_hand", "strategy": "spatial"},
                  learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])

    # model = Model(num_class=101, num_point=25, num_person=1, groups=8, block_size=41,
    #               in_channels=2, graph_args={"labeling_mode": "spatial"}, learning_rate=config["learning_rate"], 
    #               weight_decay=config["weight_decay"], step_size=5)
    
# params={'batch_size': 4, 'learning_rate': 0.00016864521947381232, 'weight_decay': 0.007578112205471038}
    data = VSLDataModule(config["batch_size"])
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]
    trainer = pl.Trainer(max_epochs = 10, accelerator="gpu", check_val_every_n_epoch = 1, 
                        devices = 1, callbacks=callbacks)
    trainer.fit(model, data)

if __name__ == '__main__':

    search_space = {
        "batch_size" : tune.choice([1,2,4]),
        "learning_rate" : tune.loguniform(1e-5, 1e-1),
        "weight_decay" : tune.loguniform(1e-5, 1e-1)
    }

    # Execute the hyperparameter search
    analysis = tune.run(
        tune.with_parameters(train_vsl),
        resources_per_trial={
            "cpu": 1,
            "gpu": 1
        },
        metric="loss",
        mode="min",
        config=search_space,
        num_samples=10,
        name="tune_vsl")

    print(analysis.best_config)

    
    

    