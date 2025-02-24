import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tgcn import ConvTemporalGraphical
from graph import Graph
from torchinfo import summary

import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
import torch.optim as optim

import numpy as np

class Model(pl.LightningModule):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, learning_rate, weight_decay, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        # self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.A = Variable(torch.from_numpy(self.graph.A.astype(np.float32)), requires_grad=False)
        # self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = self.A.size()[0]
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * self.A.size()[1])
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            # st_gcn(256, 256, kernel_size, 1, **kwargs),
            # st_gcn(256, 256, kernel_size, 1, **kwargs),
            # st_gcn(256, 512, kernel_size, 2, **kwargs),
            # st_gcn(512, 512, kernel_size, 1, **kwargs),
            # st_gcn(512, 512, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

        self.loss = nn.CrossEntropyLoss()
        self.metric = MulticlassAccuracy(num_class)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_step_loss_outputs = []
        self.validation_step_acc_outputs = []

        self.save_hyperparameters()

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        # N, M, V, C, T
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x.float())
        x = x.view(N, M, V, C, T)
        # N, M, C, T, V
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        # print(self.A.device)
        # self.A = self.A.to(self.device)
        self.A = self.A.cuda(x.get_device())
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        y_pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        # print("Targets : ", targets)
        # print("Preds : ", y_pred_class) 
        train_accuracy = self.metric(y_pred_class, targets)
        loss = self.loss(outputs, targets)
        self.log('train_accuracy', train_accuracy, prog_bar=True, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        # return {"loss": loss, "train_accuracy" : train_accuracy}
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        y_pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        valid_accuracy = self.metric(y_pred_class, targets)
        loss = self.loss(outputs, targets)
        self.log('valid_accuracy', valid_accuracy, prog_bar=True, on_epoch=True)
        self.log('valid_loss', loss, prog_bar=True, on_epoch=True)
        self.validation_step_loss_outputs.append(loss)
        self.validation_step_acc_outputs.append(valid_accuracy)
        return {"valid_loss" : loss, "valid_accuracy" : valid_accuracy}
    
    def on_validation_epoch_end(self):
        # avg_loss = torch.stack(
            # [x["valid_loss"] for x in outputs]).mean()
        # avg_acc = torch.stack(
            # [x["valid_accuracy"] for x in outputs]).mean()
        avg_loss = torch.stack(self.validation_step_loss_outputs).mean()
        avg_acc = torch.stack(self.validation_step_acc_outputs).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)
        self.validation_step_loss_outputs.clear() 
        self.validation_step_acc_outputs.clear()

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        y_pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        print("Targets : ", targets)
        print("Preds : ", y_pred_class)
        test_accuracy = self.metric(y_pred_class, targets)
        loss = self.loss(outputs, targets)
        self.log('test_accuracy', test_accuracy, prog_bar=True, on_epoch=True)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)
        return {"test_loss" : loss, "test_accuracy" : test_accuracy}

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr = self.learning_rate, weight_decay = self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
        return {"optimizer": optimizer, 
                "lr_scheduler": {"scheduler": scheduler, "monitor": "valid_accuracy"}
               }
        # return optimizer  

    def predict_step(self, batch, batch_idx):
        return self(batch)

class Model_TwoStream(pl.LightningModule):

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, learning_rate, weight_decay, **kwargs):
        super().__init__()


        self.origin_stream = Model(in_channels, num_class, graph_args,
                 edge_importance_weighting, learning_rate, weight_decay, **kwargs)
        self.motion_stream = Model(in_channels, num_class, graph_args,
                 edge_importance_weighting, learning_rate, weight_decay, **kwargs)
        self.loss = nn.CrossEntropyLoss()
        self.metric = MulticlassAccuracy(num_class)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x):
        N, C, T, V, M = x.size()
        m = torch.cat((torch.cuda.FloatTensor(N, C, 1, V, M).zero_(),
                        x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
                        torch.cuda.FloatTensor(N, C, 1, V, M).zero_()), 2)

        res = self.origin_stream(x) + self.motion_stream(m)
        return res

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        y_pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        # print("Targets : ", targets)
        # print("Preds : ", y_pred_class) 
        train_accuracy = self.metric(y_pred_class, targets)
        loss = self.loss(outputs, targets)
        self.log('train_accuracy', train_accuracy, prog_bar=True, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        # return {"loss": loss, "train_accuracy" : train_accuracy}
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        y_pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        valid_accuracy = self.metric(y_pred_class, targets)
        loss = self.loss(outputs, targets)
        self.log('valid_accuracy', valid_accuracy, prog_bar=True, on_epoch=True)
        self.log('valid_loss', loss, prog_bar=True, on_epoch=True)
        return {"valid_loss" : loss, "valid_accuracy" : valid_accuracy}

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        y_pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        print("Targets : ", targets)
        print("Preds : ", y_pred_class)
        test_accuracy = self.metric(y_pred_class, targets)
        loss = self.loss(outputs, targets)
        self.log('test_accuracy', test_accuracy, prog_bar=True, on_epoch=True)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)
        return {"test_loss" : loss, "test_accuracy" : test_accuracy}

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr = self.learning_rate, weight_decay = self.weight_decay)
        return optimizer  

    def predict_step(self, batch, batch_idx):
        return self(batch)
    
class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        # self.relu = nn.ReLU(inplace=True)
        self.elu = nn.ELU(alpha=0.8, inplace=False)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.elu(x), A
    
if __name__ == '__main__':
    # in_channels (int): Number of channels in the input data
    #     num_class (int): Number of classes for the classification task
    #     graph_args (dict): The arguments for building the graph
    #     edge_importance_weighting (bool): If ``True``, adds a learnable
    #         importance weighting to the edges of the graph
    #     **kwargs (optional): Other parameters for graph convolution units
    a = Model(3, 20, graph_args = {"layout": "mediapipe", "strategy": "spatial"}, edge_importance_weighting=True, learning_rate=0.003, weight_decay=0.001)
    summary(a, input_size=(32, 3, 80, 25, 1), col_names=["input_size", "output_size", "num_params"])
