import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 10
x = torch.unsqueeze(torch.linspace(0, 10, 20), dim=1)
y = (-1) * x + 0.5 * torch.randn(x.size()) + 10

torch_dataset = Data.TensorDataset(x, y)

# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
    num_workers=2,  # 多线程来读数据
)

net1 = torch.nn.Sequential(
    torch.nn.Linear(1, 1),
)

# class LinearRegression(torch.nn.Module):
#     def __init__(self):
#         super(LinearRegression, self).__init__()
#         self.linear = torch.nn.Linear(1, 1, bias=True)  # input and output is 1 dimension
#
#     def forward(self, x):
#         out = self.linear(x)
#         return out
#
#
# net2 = LinearRegression()

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net1.parameters(), lr=0.02)

# for epoch in range(500):
#     for step, (batch_x, batch_y) in enumerate(loader):
for i in range(1000):
    prediction = net1(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        # if step == 0 and epoch%50 ==0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy())
        plt.pause(0.1)

    # print(batch_x, prediction)
    # 打出来一些数据
    # print('Epoch: ', epoch, '| Step: ', step, '| batch x: ', batch_x.numpy(), '| batch y: ', batch_y.numpy())

# dd = [[5.],
#       [1.],
#       [9.]]
# tensor = torch.FloatTensor(dd)
#
# yy = net1(tensor)
# print(yy)
