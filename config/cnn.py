import torch


class config():
    def __init__(self):
        self.train_batch_size = 32
        self.val_batch_size = 32
        self.gray = False
        self.epochs = 15
        self.lr = 0.01
        self.momentum = 0.9
        self.weight_decay = 0.005
        self.use_gpu = True and torch.cuda.is_available()
        self.seed = 0
        self.display = 20
        self.name = '1-frame-c2d-res34'
