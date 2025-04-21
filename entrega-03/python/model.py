
import torch.nn as nn
import torch.nn.functional as F





# from ray import tune
# from ray import train
# from ray.train import Checkpoint, get_checkpoint
# from ray.tune.schedulers import ASHAScheduler, FIFOScheduler
# from torch.utils.data import random_split
# from functools import partial
# import os
# import tempfile
# from pathlib import Path
# import torch.optim as optim

# from torch.utils.data import random_split

# from functools import partial
# import ray.cloudpickle as pickle

# class CNN(nn.Module):
#     def __init__(self,
#                  kernel_size=3,
#                  stride=1,
#                  dilation=1,
#                  conv_1_filters=4,
#                  conv_2_filters=16,
#                  conv_3_filters=0,
#                  lin_dim_1=120,
#                  lin_dim_2=80,
#                  padding=0,
#                  pooling=Pooling.MAX):
#         super().__init__()

#         if pooling == Pooling.MAX:
#             self.pool = nn.MaxPool2d(2, 2)
#         elif pooling == Pooling.AVG:
#             self.pool = nn.AvgPool2d(2, 2)
#         else:
#             self.pool = nn.Identity()

#         self.conv1 = nn.Conv2d(1, conv_1_filters, kernel_size, stride=stride, dilation=dilation, padding=padding)
#         height, width = self.get_dimensions_after_layer(28, 28,
#                                                         kernel_size, stride, padding, pooling != Pooling.NONE)

#         self.conv2 = nn.Conv2d(conv_1_filters, conv_2_filters, kernel_size, stride=stride, dilation=dilation, padding=padding)
#         height, width = self.get_dimensions_after_layer(height, width,
#                                                         kernel_size, stride, padding, pooling != Pooling.NONE)

#         self.conv3 = None
#         if conv_3_filters > 0:
#             self.conv3 = nn.Conv2d(conv_2_filters, conv_3_filters,  kernel_size, stride=stride, dilation=dilation, padding=padding)
#             height, width = self.get_dimensions_after_layer(height, width,
#                                                         kernel_size, stride, padding, pooling=False)
#         final_filters = conv_3_filters if conv_3_filters > 0 else conv_2_filters

#         self.fc1 = nn.Linear(final_filters * height * width, lin_dim_1)
#         self.fc2 = nn.Linear(lin_dim_1, lin_dim_2)
#         self.fc3 = nn.Linear(lin_dim_2, 10)

#         self.features = [
#             self.conv1,
#             self.pool,
#             self.conv2,
#             self.conv3
#         ]

#     def get_dimensions_after_layer(self, height, width, kernel, stride, padding, pooling=True):
#         height = ((height - kernel + 2 * padding) // stride) + 1
#         width = ((width - kernel + 2 * padding) // stride) + 1
#         if pooling:
#             height //= 2
#             width //= 2
#         return height, width


#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         if self.conv3:
#             x = F.relu(self.conv3(x))
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

input_size = (28, 28)
conv_kernel_size = (7, 7)
conv_filter_num = 4
pool_size = (2, 2)
dense_size = 10
training_epochs = 6
pad = conv_kernel_size[0] // 2

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(1, conv_filter_num, conv_kernel_size, padding=pad)
        self.pool = nn.MaxPool2d(pool_size)
        self.flattened_size = conv_filter_num * (input_size[0] // pool_size[0]) * (input_size[1] // pool_size[1])
        self.fc = nn.Linear(self.flattened_size, dense_size)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)