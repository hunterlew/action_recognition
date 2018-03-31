from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from torch.utils.data import DataLoader
from data.action_data import ucf_50
from model.cnn import *
from solver.common import common_solver
from config.cnn import config
from utils.visualize import *


# loading configuration
cfg = config()
if cfg.use_gpu:
    torch.cuda.set_device(1)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
else:
    torch.manual_seed(cfg.seed)

    
# loading dataset
image_ready = 1
train_loader = DataLoader(ucf_50(0, 224, image_ready), cfg.train_batch_size, True)
val_loader = DataLoader(ucf_50(1, 224, image_ready), cfg.val_batch_size, True)

# loading net
# model = c2d_alexnet()
# model = c2d_vgg16()
# model = c2d_vgg16_avg()
# model = c2d_googlenet_v1()
model = c2d_googlenet_v2()
if cfg.use_gpu:
    model.cuda()

# loading solver
optim = common_solver(train_loader, val_loader, model, cfg)

# main process
x = []
y1 = []
y2 = []
y3 = []
y4 = []
for epoch in range(1, cfg.epochs + 1):
    optim.train(epoch)
    train_loss, train_accuracy, val_loss, val_accuracy = optim.val()
    x += [epoch]
    y1 += [train_loss]
    y2 += [val_loss]
    y3 += [train_accuracy]
    y4 += [val_accuracy]

draw_curves(x, y1, y2, y3, y4, cfg.name)  # draw the curves
