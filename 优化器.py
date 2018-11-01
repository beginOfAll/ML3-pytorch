import torch
from torch import nn as torch_nn
from torch import optim as torch_optim
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.randn(x.size())
# y = x.pow(2) + .1 * torch.normal(torch.zeros(*x.size()), torch.ones(*x.size()))
# plot dataset
plt.scatter(x.numpy(), y.numpy())
plt.show()

torch_dataset = Data.TensorDataset(x, y)

# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
    num_workers=2,  # 多线程来读数据
)


# net1 = torch_nn.Sequential(
#     torch_nn.Linear(1, 1),
# )

class Net(torch_nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch_nn.Linear(n_feature, n_hidden)
        self.predict = torch_nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


# 为每个优化器创建一个 net
net_SGD = Net(1, 20, 1)
net_Momentum = Net(1, 20, 1)
net_RMSprop = Net(1, 20, 1)
net_Adam = Net(1, 20, 1)
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]


# different optimizers
opt_SGD = torch_optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum = torch_optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop = torch_optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam = torch_optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = torch_nn.MSELoss()
losses_his = [[], [], [], []]   # 记录 training 时不同神经网络的 loss


for epoch in range(EPOCH):
    print('Epoch: ', epoch)
    for step, (b_x, b_y) in enumerate(loader):
        b_x = Variable(b_x)
        b_y = Variable(b_y)
        # 对每个优化器, 优化属于他的神经网络
        for net, opt, l_his in zip(nets, optimizers, losses_his):
            output = net(b_x)  # get output for every net
            loss = loss_func(output, b_y)  # compute loss for every net
            opt.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            opt.step()  # apply gradients
            l_his.append(loss.data.numpy())  # loss recoder

labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()
