import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from utils.graph_conv import calculate_laplacian_with_self_loop
from utils.losses import mse_with_regularizer_loss
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy.linalg as la
import math

###### evaluation ######
def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b)/la.norm(a)
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a - b))/np.var(a)
    return rmse, mae, 1-F_norm, r2, var

def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    print('feat.shape: ', feat.shape)
    return feat


def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    print('adj.shape: ', adj.shape)
    return adj


def generate_dataset(data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True):
    """
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        data = data / max_val
    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:time_len]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i:i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len:i + seq_len + pre_len]))
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i:i + seq_len]))
        test_Y.append(np.array(test_data[i + seq_len:i + seq_len + pre_len]))
    return np.array(train_X), np.array(train_Y), \
        np.array(test_X), np.array(test_Y)


def generate_torch_datasets(data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True):
    train_X, train_Y, test_X, test_Y = generate_dataset(data, seq_len, pre_len, time_len=time_len,
                                                        split_ratio=split_ratio, normalize=normalize)
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_X), torch.FloatTensor(train_Y))
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(test_X), torch.FloatTensor(test_Y))
    return train_dataset, test_dataset

class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.register_buffer('laplacian', calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
        self.weights = nn.Parameter(torch.FloatTensor(self._num_gru_units + 1, self._output_dim))
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        hidden_state = hidden_state.reshape((batch_size, num_nodes, self._num_gru_units))
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        concatenation = concatenation.reshape((num_nodes, (self._num_gru_units + 1) * batch_size))
        a_times_concat = self.laplacian @ concatenation
        a_times_concat = a_times_concat.reshape((num_nodes, self._num_gru_units + 1, batch_size))
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        a_times_concat = a_times_concat.reshape((batch_size * num_nodes, self._num_gru_units + 1))
        outputs = a_times_concat @ self.weights + self.biases
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            'num_gru_units': self._num_gru_units,
            'output_dim': self._output_dim,
            'bias_init_value': self._bias_init_value
        }


class TGCNCell(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.register_buffer('adj', torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolution(self.adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0)
        self.graph_conv2 = TGCNGraphConvolution(self.adj, self._hidden_dim, self._hidden_dim)

    def forward(self, inputs, hidden_state):
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {
            'input_dim': self._input_dim,
            'hidden_dim': self._hidden_dim
        }


class TGCN(nn.Module):
    def __init__(self, adj, hidden_dim: int, **kwargs):
        super(TGCN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.register_buffer('adj', torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCell(self.adj, self._input_dim, self._hidden_dim)
        # 所以FC Layer默认的Bias是设置的
        self.regressor = nn.Linear(hidden_dim, 3, bias=True)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(inputs)
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))

        # 目前output指的是我们TGCN架构的输出结果

        # 这里很重要，我们在regressor完成一次forward propagation
        predictions = self.regressor(output)  # 此时[128, 15, 3]

        # 然后我们把predictions重新map到[batch_size, pre_len, num_nodes]这样的维度
        predictions = predictions.permute(0, 2, 1)

        return predictions

        # predictions作为全连接层的输出


    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {
            'input_dim': self._input_dim,
            'hidden_dim': self._hidden_dim
        }

# 数据加载器能够使用的进程数量
workers = 2

# 训练时的批大小
batch_size = 128

# 训练次数
num_epochs = 500

# 优化器学习率
# lr = 0.0002
lr = 0.001

# Adam优化器的Beta1超参
beta1 = 0.5

# 可利用的GPU数量，使用0将运行在CPU模式
ngpu = 1

"""
===================================  我们先处理数据  ===========================================
"""
# 我们先读取adjacency_matrix和feature
adj = load_adjacency_matrix('data/hz_adj.csv')
feat = load_features('data/hz_speed.csv')

# 然后我们读取dataset
train_dataset, test_dataset = generate_torch_datasets(feat, 12, 3, None, 0.8, True)

# 这里我们用到dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# 实例化model
model = TGCN(adj=adj, hidden_dim=64).to(device)
model.load_state_dict(torch.load('tgcn.pth'))

# Test阶段
model.eval()
y_list = []
pred_list = []
for i, data in enumerate(test_dataloader):
    his_speed = data[0].float().to(device)
    y_speed = data[1].float().to(device)
    pred_speed = model(his_speed)
    y_list.append(y_speed)
    pred_list.append(pred_speed)

y = torch.cat(y_list, dim=0)
pred = torch.cat(pred_list, dim=0)

# 此时，y和pred都为[2865, 3, 15]

# 取子集
y = y[:, :, [0,4,5,13]]
pred = pred[:, :, [0,4,5,13]]

y = torch.flatten(y, start_dim=0, end_dim=1)
pred = torch.flatten(pred, start_dim=0, end_dim=1)

np_y = y.cpu().detach().numpy()
np_pred = pred.cpu().detach().numpy()

rmse, mae, accuracy, r2, var = evaluation(np_y, np_pred)
print('rmse:%r\n' % rmse,
      'mae:%r\n' % mae,
      'acc:%r\n' % accuracy,
      'r2:%r\n' % r2,
      'var:%r\n' % var)