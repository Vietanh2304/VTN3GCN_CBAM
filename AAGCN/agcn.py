import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from graph import Graph
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
import torch.optim as optim

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Model(pl.LightningModule):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 learning_rate = 1e-4, weight_decay = 1e-4):
        super(Model, self).__init__()

        # if graph is None:
        #     raise ValueError()
        # else:
        # Graph = import_class(graph)
        self.graph = Graph(**graph_args)

        A = self.graph.A
        # print(num_person * in_channels * num_point)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(in_channels, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

        self.loss = nn.CrossEntropyLoss()
        self.metric = MulticlassAccuracy(num_class)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_step_loss_outputs = []
        self.validation_step_acc_outputs = []

        self.save_hyperparameters()

    def forward(self, x):
        # 0, 1, 2, 3, 4 
        N, C, T, V, M = x.size()
        # print(f"N {N}, C {C}, T {T}, V {V}, M {M}")
        # N, M, V, C, T
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        # print(M*V*C)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)
    
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

if __name__ == "__main__":  
    import os
    from torchinfo import summary
    print(os.getcwd())  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(num_class=20, num_point=25, num_person=1, 
                  graph_args={"layout":"mediapipe", "strategy":"spatial"}, in_channels=2).to(device)
    # print(model.device)
    # N, C, T, V, M
    summary(model)
    x = torch.randn((1, 2, 80, 25, 1)).to(device)
    y = model(x)    
    print(y.shape)
