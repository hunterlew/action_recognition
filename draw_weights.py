import matplotlib
matplotlib.use('Agg')
from model.cnn import *
import torch
from utils.visualize import *

# model = c2d_alexnet()
# model.load_state_dict(torch.load('save/1-frame-c2d-alex/net-epoch-15.pkl'))
model = c2d_googlenet()
model.load_state_dict(torch.load('save/1-frame-c2d-google/net-epoch-15.pkl'))

c1 = model.conv1.weight.data.numpy()
print(np.shape(c1))
draw_feature(c1.transpose(0, 2, 3, 1), '1-frame-c2d-google')
