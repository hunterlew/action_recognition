import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# define optimizer
class common_solver():
    def __init__(self, t_ld, v_ld, model, config):
        self.train_loader = t_ld
        self.val_loader = v_ld
        self.model = model
        self.config = config
        self.optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
        # self.optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # define training and testing process
    def train(self, epoch):
        self.model.train()  # switch to training process
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.config.use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            # step learning policy
            adjust_lr = self.config.lr * (0.1 ** (epoch // 10))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = adjust_lr
            self.optimizer.zero_grad()

            if self.config.gray:
                output = self.model(data[:, :1, :, :])
            else:
                output = self.model(data)

            loss = F.nll_loss(output, target)  # the negative log likelihood loss, in average by default
            loss.backward()
            self.optimizer.step()  # update
            if batch_idx % self.config.display == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.data[0]))
            if not os.path.exists('save/' + self.config.name):
                os.mkdir('save/' + self.config.name)
            torch.save(self.model.state_dict(), 'save/' + self.config.name + '/net-epoch-' + str(epoch) + '.pkl')

    def val(self):
        self.model.eval()  # switch to evaluation process

        # evaluate on the train set
        train_loss = 0
        train_correct = 0
        for data, target in self.train_loader:
            if self.config.use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)

            if self.config.gray:
                output = self.model(data[:, :1, :, :])
            else:
                output = self.model(data)
            train_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            train_correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        train_loss /= len(self.train_loader.dataset)
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            train_loss, train_correct, len(self.train_loader.dataset),
            100. * train_correct / len(self.train_loader.dataset)))

        # evaluate on the val set
        val_loss = 0
        val_correct = 0
        for data, target in self.val_loader:
            if self.config.use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)

            if self.config.gray:
                output = self.model(data[:, :1, :, :])
            else:
                output = self.model(data)
            val_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            val_correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        val_loss /= len(self.val_loader.dataset)
        print('Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            val_loss, val_correct, len(self.val_loader.dataset),
            100. * val_correct / len(self.val_loader.dataset)))

        return train_loss, 100. * train_correct / len(self.train_loader.dataset), val_loss, 100. * val_correct / len(self.val_loader.dataset)
