from __future__ import print_function
#%matplotlib inline

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
from IPython.display import HTML
from utils.graph_conv import calculate_laplacian_with_self_loop
from utils.losses import mse_with_regularizer_loss
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy.linalg as la
import argparse

def generate_matrix_multi_slots(the_inputs, b_adj):

    ### 我们这里讨论的是针对多个time slots的matrix generation操作，也就是针对的(128, 13, 15)!
    ### 注意，我们这里是先concat到多个time slots上，然后再生成相应的matrix

    _, seq_len, _ = the_inputs.shape
    traffic_matrices = []
    for i in range(seq_len):
        the_input = torch.unsqueeze(the_inputs[:, i, :], dim=1)
        # print('the_input.size: ', list(the_input.size()))

        traffic_matrix = generate_matrix_one_slot(the_input, b_adj)
        traffic_matrices.append(torch.unsqueeze(traffic_matrix, dim=1))


    ## traffic_matrices包含了13个(128, 1, 15, 15)

    traffic_matrices_multi_slots = torch.cat(traffic_matrices, dim=1)
    # print('traffic_matrices_multi_slots.size: ', list(traffic_matrices_multi_slots.size()))
    # input()

    return traffic_matrices_multi_slots


# Customized Function for Replicate/Transpose/Element_Wise_Multiple
def generate_matrix_one_slot(the_input, b_adj):

    ### 我们这里讨论的是针对一个time slot的matrix generation操作，也就是针对(128, 1, 15)
    ### the_input (128, 1, 15)
    ### b_adj (128, 15, 15)

    # 1. 完成Replicate
    the_input = the_input.repeat(1, 15, 1)

    # 2. 完成Transpose
    the_input = torch.transpose(the_input, 1, 2)

    # 3. 完成Element_Wise_Multiple
    the_input = the_input * b_adj

    return the_input
    # 注意这个the_input是 (128, 15, 15)


###### evaluation ######
def evaluation(a, b):
    rmse = math.sqrt(mean_squared_error(a, b))
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
        print('max_val: ', max_val)
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
    test_dataset  = torch.utils.data.TensorDataset(torch.FloatTensor(test_X), torch.FloatTensor(test_Y))

    print('train_X.size: ', train_X.shape)
    print('train_Y.size: ', train_Y.shape)
    print('test_X.size: ', test_X.shape)
    print('test_Y.size: ', test_Y.shape)

    # print('train_X: ', train_X)
    # print('train_Y: ', train_Y)

    # input()

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


# 生成器代码
class TGCN(nn.Module):
    def __init__(self, adj, hidden_dim: int, **kwargs):
        super(TGCN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.register_buffer('adj', torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCell(self.adj, self._input_dim, self._hidden_dim)
        # 所以FC Layer默认的Bias是设置的
        self.regressor = nn.Linear(hidden_dim, 1, bias=True)

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
        predictions = self.regressor(output)  # 此时是(128, 15, 1)

        # 然后经过维度交换
        predictions = predictions.permute(0, 2, 1)  # 此时是(128, 1, 15)

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

class D_CNN(nn.Module):
    def __init__(self):
        super(D_CNN, self).__init__()
        self.main = nn.Sequential(
            # 输入大小 (batch_size, 1, 15, 15)
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 输入大小 (batch_size, 8, 8, 8)
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(0.2, inplace=True),

            # 输入大小 (batch_size, 16, 4, 4)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(0.2, inplace=True),

            # 输入大小 (batch_size, 32, 2, 2)
            # 然后我们把数据Flatten到128维度的向量
            nn.Flatten()

            # 输出大小 (batch_size, 128)
        )

    def forward(self, input):
        return self.main(input)

class D_LSTM(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, bi_dir=False):
        super(D_LSTM, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, bidirectional=False)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        return out, h

# 判别器代码
class Discriminator(nn.Module):
    def __init__(self, ngpu, h_dim):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.d_cnn = D_CNN()
        self.d_lstm = D_LSTM(input_dim=128, hidden_dim=h_dim, bi_dir=False)

        # Self Attention的计算用到了下面两个Linear层
        self.linear1 = nn.Linear(in_features=64, out_features=1, bias=True)
        # 然后是Softmax函数
        self.soft = nn.Softmax(dim=1)

        self.FC = nn.Linear(in_features=64, out_features=1, bias=True)

    def forward(self, inputs, h_dim):
        b_size, seq_len, height, width = inputs.shape
        # 理论上来讲inputs的shape是(128, 13, 15, 15)

        # Step 1. 我们让每一个timestep经过一遍CNN

        conv_results = []
        for i in range(seq_len):
            one_slot_pic = torch.unsqueeze(inputs[:, i, :, :], dim=1)   # 理论上one_slot_pic应该是(128, 1, 15, 15)
            conv_result = self.d_cnn(one_slot_pic)   # 理论上conv_result应该是(128, 128)
            conv_result = torch.unsqueeze(conv_result, dim=1)  # 理论上conv_result变成(128, 1, 128)
            conv_results.append(conv_result)

        # Step 2. 现在我们有13个(128, 1, 128)这样的Tensor

        x = torch.cat(conv_results, dim=1)  # 理论上x应该是(128, 13, 128)
        h = torch.zeros(1, seq_len, h_dim).to(device)
        out, h = self.d_lstm(x, h)

        # 现在out的维度应该是(128, 13, 64)  注意 Torch返回了每一个时刻的output
        e_matrix = self.linear1(out)  # e_matrix为(128, 13, 1)
        attn_matrix = self.soft(e_matrix)  # attn_matrix为(128, 13, 1)

        # 计算C_matrix(注意这里用到了PyTorch里面的broadcast!!!!)
        C_matrix = attn_matrix * out  # 此时C_matrix为(128, 13, 64)

        # 然后对C_matrix在dim=1的维度进行sum运算
        C_matrix = torch.sum(C_matrix, dim=1)  # 理论上C_matrix应该是(128, 1, 64)

        # 然后我们最后再通过一个FC层将out转换成(128, 1)这样子
        out = self.FC(C_matrix)

        # 没有sigmoid!!!

        return out

# 为了可重复性设置随机种子
manualSeed = 1222
#manualSeed = random.randint(1, 10000) # 如果你想有一个不同的结果使用这行代码
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

"""
==========================================================================================
"""

# 数据加载器能够使用的进程数量
workers = 2

# 训练时的批大小
batch_size = 128

# 训练次数
num_epochs = 100

# 优化器学习率
lr = 0.0002

# Adam优化器的Beta1超参
beta1 = 0.5

# 可利用的GPU数量，使用0将运行在CPU模式
ngpu = 1

"""
==========================================================================================
"""

# 我们先读取adjacency_matrix和feature
adj = load_adjacency_matrix('data/hz_adj.csv')
feat = load_features('data/hz_speed.csv')

# 现在只有train_dataset和test_dataset!
train_dataset, test_dataset = generate_torch_datasets(data=feat, seq_len=12, pre_len=1, time_len=None, split_ratio=0.8, normalize=True)

# 现在只用这两个dataloader!
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

# 决定我们在哪个设备上运行
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# 在netG和netD上调用的自定义权重初始化函数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

"""
==========================================================================================
"""


# 创建生成器
netG = TGCN(adj=adj, hidden_dim=64).to(device)
# 输出该模型
print(netG)

parser = argparse.ArgumentParser(description='GAN')
parser.add_argument("--pretrain_epoch", type=int, default=200, help="pretrained model state")
args = parser.parse_args()

# 初始化netG的参数, 我们直接用预训练的结果好了
netG.load_state_dict(torch.load('pretrained/pretrained_G_' + str(args.pretrain_epoch) + '.pth'))

# 创建判别器
netD = Discriminator(ngpu=ngpu, h_dim=64).to(device)

# 使用权重初始化函数 weights_init 去随机初始化所有权重
#  mean=0, stdev=0.2.
netD.apply(weights_init)

# 输出该模型
print(netD)

"""
==========================================================================================
"""


"""
==========================================================================================
"""

# 对了，我们先用手写loss的方式实现WGAN，我们不需要让D的输出去和label做比较
# criterion = nn.MSELoss()

# 建立一个在训练中使用的真实和假的标记 (这里是尤其容易翻车的地方，要想清楚)
real_label = 1
fake_label = -1

# 为G和D都设置RMSprop优化器
optimizerD = optim.RMSprop(netD.parameters(), lr=0.0001)
optimizerG = optim.RMSprop(netG.parameters(), lr=0.0001)

"""
==========================================================================================
"""
# 训练循环

# 保存跟踪进度的列表
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop (Pretrain_G epoch " + str(args.pretrain_epoch) + ")...")
# 我们把邻接矩阵拿出来(注意要变成(128, 15, 15))
adj_tensor = torch.tensor(adj).to(device)            #  adj_tensor 维度是 (15, 15)
adj_tensor = torch.unsqueeze(adj_tensor, dim=0)      #  adj_tensor 维度是 (1, 15, 15)

best_epoch = -100
best_acc = -100

# 每个epoch
for epoch in range(num_epochs):  # 遍历每一个epoch

    netG.train()  # 在每一个epoch的开头位置, 我们都强制Generator位于训练模式

    for i, data in enumerate(train_dataloader, 0):  # 遍历每一个mini_batch

        # 注意data分为2部分：
        # data[0]为(128, 12, 15), 也就是前12个时刻的历史数据
        # data[1]为(128, 1,  15), 也就是第13个时刻的真实数据
        data_0 = data[0].float().to(device)
        data_1 = data[1].float().to(device)

        real = data_1.to(device)  # 现在real维度是(128, 1, 15)
        b_size = real.size(0)  # 注意b_size可能小于128

        # 生成关于该batch的adjacency matrix
        b_adj = adj_tensor.repeat(b_size, 1, 1)  # 注意b_adj需要在知道b_size之后才能生成, b_size可能小于128

        avg_real_score = -1
        avg_fake_score = -1
        avg_cheat_score = -1

        ############################
        # (1) 更新 D 网络:
        ############################

        # 我们先拼出(128, 13, 15)的real样本
        real = torch.cat([data_0, data_1], dim=1)

        # 然后我们生成(128, 13, 15, 15)的traffic matrix
        real = generate_matrix_multi_slots(real, b_adj)

        # traffic network matrix -> netD -> score (that can be directly used to compute loss)

        netD.zero_grad()
        netG.zero_grad()
        ## 使用Real Sample进行训练

        # 通过D对real数据做forward propagation, 我们得到score_real
        score_real = netD(real, h_dim=64).view(-1)
        avg_real_score = torch.mean(score_real).item()

        ## 使用Fake Sample进行训练

        # 使用生成器G生成pred
        fake = netG(data_0).to(device)  # netG接收(128, 12, 15)输出(128, 1, 15)
        # 用(128, 12, 15)和(128, 1, 15)拼出(128, 13, 15)
        fake = torch.cat([data_0, fake], dim=1)
        # 然后我们生成(128, 13, 15, 15)的traffic matrix
        fake = generate_matrix_multi_slots(fake, b_adj)
        # 计算score_fake
        score_fake = netD(fake, h_dim=64).view(-1)
        avg_fake_score = torch.mean(score_fake).item()

        # 使用正确的Wasserstein Loss!!!
        errD = - (torch.mean(score_real) - torch.mean(score_fake))
        errD.backward()

        # print('torch.mean(real): ', torch.mean(output_real).item(), 'torch.mean(fake): ', torch.mean(output_fake).item())

        # 更新判别器D
        optimizerD.step()

        # Discriminator网络的参数裁剪！
        for p in netD.parameters():
            p.data.clamp_(-0.005, 0.005)

        if i % 10 == 0:

            ############################
            # (2) 更新 G 网络:
            ############################
            netG.zero_grad()
            netD.zero_grad()

            fake2 = netG(data_0).to(device)   # fake2是(128, 1, 15)
            fake2 = torch.cat([data_0, fake2], dim=1)  # (128, 12, 15) + (128, 1, 15) -> (128, 13, 15)
            fake2 = generate_matrix_multi_slots(fake2, b_adj)

            # 因为我们之前更新了D，通过D执行所有假样本批次的正向传递
            score_cheat = netD(fake2, h_dim=64).view(-1)

            # 基于这个输出计算G的损失，使用Wasserstein Loss
            errG = -torch.mean(score_cheat)
            # 为生成器计算梯度
            errG.backward()
            # 更新生成器G
            optimizerG.step()
            avg_cheat_score = errG.item()

        # 输出训练状态
        """
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\terrD: %.4f\terrG: %.4f\tReal: %.4f\tFake: %.4f\tCheat: %.4f'
                  % (epoch, num_epochs, i, len(train_dataloader),
                     errD.item() * 1000000, errG.item() * 1000000, avg_real_score * 1000000, avg_fake_score * 1000000, avg_cheat_score * 1000000))
        """
        # 为以后画损失图，保存损失
        G_losses.append(errG.item())  # errG只有1项
        D_losses.append(errD.item())  # errD包括errD_real和errD_fake

        iters += 1

    ###############################################
    # 在每一个epoch之后，我们都进行测试，不放过任何一个epoch
    ###############################################
    netG.eval()  # 在每一个epoch的结束位置，我们将Generator切换到推理模式
    y_list = []
    pred_list = []
    for i, data in enumerate(test_dataloader):
        his_speed = data[0].float().to(device)  # (128, 12, 15)
        y_speed = data[1].float().to(device)  # (128, 1, 15)
        pred_speed = netG(his_speed)  # (128, 1, 15)
        y_list.append(y_speed)
        pred_list.append(pred_speed)

    y = torch.cat(y_list, dim=0)
    pred = torch.cat(pred_list, dim=0)

    y = torch.flatten(y, start_dim=0, end_dim=1)
    pred = torch.flatten(pred, start_dim=0, end_dim=1)

    np_y = y.cpu().detach().numpy()
    np_pred = pred.cpu().detach().numpy()

    rmse, mae, accuracy, r2, var = evaluation(np_y, np_pred)
    print('epoch: ' + str(epoch) + '|| acc: %.4f' % accuracy)

    if accuracy > best_acc:
        best_acc = accuracy
        best_epoch = epoch
print('best_epoch: ', best_epoch)
print('best_acc: %.4f' % best_acc)

"""
==========================================================================================
"""

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('WGAN loss.png')

"""
保存一下模型
"""

# 保存训练好的模型
torch.save(netG.state_dict(), 'netG.pth')

"""
# 加载训练好的模型
model = TGCN(adj=adj, hidden_dim=64).to(device)
model.load_state_dict(torch.load('tgcn.pth'))
model.eval()
"""