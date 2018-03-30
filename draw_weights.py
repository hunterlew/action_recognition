import matplotlib
matplotlib.use('Agg')
from model.cnn import *
import torch
from utils.visualize import *

model = c2d_alexnet()
model.load_state_dict(torch.load('save/1-frame-c2d-alexnet/net-epoch-13.pkl'))
# model = c2d_vgg16_avg()
# model.load_state_dict(torch.load('net-epoch-15.pkl'))

c1 = model.conv1.weight.data.numpy()
print(np.shape(c1))
draw_feature(c1.transpose(0, 2, 3, 1), '1-frame-c2d-alexnet')
