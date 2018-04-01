import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import models

# define net
class c2d_alexnet(nn.Module):
    def __init__(self):
        super(c2d_alexnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(9216, 2048)  # 6*6*256 = 9216
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 50)
        self.conv1_bn = nn.BatchNorm2d(96)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3_bn = nn.BatchNorm2d(384)
        self.conv4_bn = nn.BatchNorm2d(384)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.fc1_bn = nn.BatchNorm1d(2048)
        self.fc2_bn = nn.BatchNorm1d(2048)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.max_pool2d(x, 3, stride=2)
        # print(x.shape)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.max_pool2d(x, 3, stride=2)
        # print(x.shape)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.max_pool2d(x, 3, stride=2)
        # print(x.shape)

        x = x.view(-1, 9216)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        # print(x.shape)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
        return F.log_softmax(x, dim=1)


class c2d_vgg16(nn.Module):
    def __init__(self):
        super(c2d_vgg16, self).__init__()
        self.conv1_a = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_a = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_c = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_a = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_b = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_c = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_a = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_b = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_c = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(512, 50)
        self.conv1_a_bn = nn.BatchNorm2d(64)
        self.conv1_b_bn = nn.BatchNorm2d(64)
        self.conv2_a_bn = nn.BatchNorm2d(128)
        self.conv2_b_bn = nn.BatchNorm2d(128)
        self.conv3_a_bn = nn.BatchNorm2d(256)
        self.conv3_b_bn = nn.BatchNorm2d(256)
        self.conv3_c_bn = nn.BatchNorm2d(256)
        self.conv4_a_bn = nn.BatchNorm2d(512)
        self.conv4_b_bn = nn.BatchNorm2d(512)
        self.conv4_c_bn = nn.BatchNorm2d(512)
        self.conv5_a_bn = nn.BatchNorm2d(512)
        self.conv5_b_bn = nn.BatchNorm2d(512)
        self.conv5_c_bn = nn.BatchNorm2d(512)

    def forward(self, x):
        x = F.relu(self.conv1_a_bn(self.conv1_a(x)))
        x = F.relu(self.conv1_b_bn(self.conv1_b(x)))
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = F.relu(self.conv2_a_bn(self.conv2_a(x)))
        x = F.relu(self.conv2_b_bn(self.conv2_b(x)))
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = F.relu(self.conv3_a_bn(self.conv3_a(x)))
        x = F.relu(self.conv3_b_bn(self.conv3_b(x)))
        x = F.relu(self.conv3_c_bn(self.conv3_c(x)))
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = F.relu(self.conv4_a_bn(self.conv4_a(x)))
        x = F.relu(self.conv4_b_bn(self.conv4_b(x)))
        x = F.relu(self.conv4_c_bn(self.conv4_c(x)))
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = F.relu(self.conv5_a_bn(self.conv5_a(x)))
        x = F.relu(self.conv5_b_bn(self.conv5_b(x)))
        x = F.relu(self.conv5_c_bn(self.conv5_c(x)))
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 512)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        # print(x.shape)
        return F.log_softmax(x, dim=1)


class inception(nn.Module):
    def __init__(self, input, a, b_1, b, c_1, c, d):  # 1conv-3conv-5conv-pool
        super(inception, self).__init__()
        self.conv_a = nn.Conv2d(input, a, kernel_size=1, stride=1, padding=0)

        self.conv_b_1 = nn.Conv2d(input, b_1, kernel_size=1, stride=1, padding=0)
        self.conv_b_2 = nn.Conv2d(b_1, b, kernel_size=3, stride=1, padding=1)

        self.conv_c_1 = nn.Conv2d(input, c_1, kernel_size=1, stride=1, padding=0)
        self.conv_c_2 = nn.Conv2d(c_1, c, kernel_size=3, stride=1, padding=1)
        self.conv_c_3 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1)

        self.conv_d = nn.Conv2d(input, d, kernel_size=1, stride=1, padding=0)

        self.conv_a_bn = nn.BatchNorm2d(a)
        self.conv_b_1_bn = nn.BatchNorm2d(b_1)
        self.conv_b_2_bn = nn.BatchNorm2d(b)
        self.conv_c_1_bn = nn.BatchNorm2d(c_1)
        self.conv_c_2_bn = nn.BatchNorm2d(c)
        self.conv_c_3_bn = nn.BatchNorm2d(c)
        self.conv_d_bn = nn.BatchNorm2d(d)
        self.output = a + b + c + d

    def forward(self, x):
        x_a = F.relu(self.conv_a_bn(self.conv_a(x)))
        x_b = F.relu(self.conv_b_1_bn(self.conv_b_1(x)))
        x_b = F.relu(self.conv_b_2_bn(self.conv_b_2(x_b)))
        x_c = F.relu(self.conv_c_1_bn(self.conv_c_1(x)))
        x_c = F.relu(self.conv_c_2_bn(self.conv_c_2(x_c)))
        x_c = F.relu(self.conv_c_3_bn(self.conv_c_3(x_c)))
        x_d = F.max_pool2d(x, 3, stride=1, padding=1)
        x_d = F.relu(self.conv_d_bn(self.conv_d(x_d)))
        x = torch.cat([x_a, x_b, x_c, x_d], dim=1)
        assert x.shape[1] == self.output
        return x


class c2d_googlenet(nn.Module):
    def __init__(self):
        super(c2d_googlenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)

        self.in1_1 = inception(192, 64, 96, 128, 16, 32, 32)
        self.in1_2 = inception(256, 128, 128, 192, 32, 96, 64)

        self.in2_1 = inception(480, 192, 96, 208, 16, 48, 64)
        self.in2_2 = inception(512, 160, 112, 224, 24, 64, 64)
        self.in2_3 = inception(512, 128, 128, 256, 24, 64, 64)
        self.in2_4 = inception(512, 112, 144, 288, 32, 64, 64)
        self.in2_5 = inception(528, 256, 160, 320, 32, 128, 128)

        self.in3_1 = inception(832, 256, 160, 320, 32, 128, 128)
        self.in3_2 = inception(832, 384, 192, 384, 48, 128, 128)

        self.fc = nn.Linear(1024, 50)

        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3_bn = nn.BatchNorm2d(192)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        # print(x.shape)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        # print(x.shape)
        x = self.in1_1(x)
        x = self.in1_2(x)
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        # print(x.shape)
        x = self.in2_1(x)
        x = self.in2_2(x)
        x = self.in2_3(x)
        x = self.in2_4(x)
        x = self.in2_5(x)
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        # print(x.shape)
        x = self.in3_1(x)
        x = self.in3_2(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        # print(x.shape)
        return F.log_softmax(x, dim=1)


class inception_v3(nn.Module):
    def __init__(self, input, a, b_1, b, c_1, c, d):
        super(inception_v3, self).__init__()
        self.conv_a = nn.Conv2d(input, a, kernel_size=1, stride=1, padding=0)

        self.conv_b_1 = nn.Conv2d(input, b_1, kernel_size=1, stride=1, padding=0)
        self.conv_b_2 = nn.Conv2d(b_1, b, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_b_3 = nn.Conv2d(b, b, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.conv_c_1 = nn.Conv2d(input, c_1, kernel_size=1, stride=1, padding=0)
        self.conv_c_2 = nn.Conv2d(c_1, c, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_c_3 = nn.Conv2d(c, c, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.conv_c_4 = nn.Conv2d(c, c, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_c_5 = nn.Conv2d(c, c, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.conv_d = nn.Conv2d(input, d, kernel_size=1, stride=1, padding=0)

        self.conv_a_bn = nn.BatchNorm2d(a)
        self.conv_b_1_bn = nn.BatchNorm2d(b_1)
        self.conv_b_2_bn = nn.BatchNorm2d(b)
        self.conv_b_3_bn = nn.BatchNorm2d(b)
        self.conv_c_1_bn = nn.BatchNorm2d(c_1)
        self.conv_c_2_bn = nn.BatchNorm2d(c)
        self.conv_c_3_bn = nn.BatchNorm2d(c)
        self.conv_c_4_bn = nn.BatchNorm2d(c)
        self.conv_c_5_bn = nn.BatchNorm2d(c)
        self.conv_d_bn = nn.BatchNorm2d(d)
        self.output = a + b + c + d

    def forward(self, x):
        x_a = F.relu(self.conv_a_bn(self.conv_a(x)))
        x_b = F.relu(self.conv_b_1_bn(self.conv_b_1(x)))
        x_b = F.relu(self.conv_b_2_bn(self.conv_b_2(x_b)))
        x_b = F.relu(self.conv_b_3_bn(self.conv_b_3(x_b)))
        x_c = F.relu(self.conv_c_1_bn(self.conv_c_1(x)))
        x_c = F.relu(self.conv_c_2_bn(self.conv_c_2(x_c)))
        x_c = F.relu(self.conv_c_3_bn(self.conv_c_3(x_c)))
        x_c = F.relu(self.conv_c_4_bn(self.conv_c_4(x_c)))
        x_c = F.relu(self.conv_c_5_bn(self.conv_c_5(x_c)))
        x_d = F.max_pool2d(x, 3, stride=1, padding=1)
        x_d = F.relu(self.conv_d_bn(self.conv_d(x_d)))
        x = torch.cat([x_a, x_b, x_c, x_d], dim=1)
        assert x.shape[1] == self.output
        return x


class inception_v3_exp(nn.Module):
    def __init__(self, input, a, b_1, b, c_1, c, d):
        super(inception_v3_exp, self).__init__()
        self.conv_a = nn.Conv2d(input, a, kernel_size=1, stride=1, padding=0)

        self.conv_b_1 = nn.Conv2d(input, b_1, kernel_size=1, stride=1, padding=0)
        self.conv_b_2 = nn.Conv2d(b_1, b, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_b_3 = nn.Conv2d(b_1, b, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.conv_c_1 = nn.Conv2d(input, c_1, kernel_size=1, stride=1, padding=0)
        self.conv_c_2 = nn.Conv2d(c_1, c, kernel_size=3, stride=1, padding=1)
        self.conv_c_3 = nn.Conv2d(c, c, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_c_4 = nn.Conv2d(c, c, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.conv_d = nn.Conv2d(input, d, kernel_size=1, stride=1, padding=0)

        self.conv_a_bn = nn.BatchNorm2d(a)
        self.conv_b_1_bn = nn.BatchNorm2d(b_1)
        self.conv_b_2_bn = nn.BatchNorm2d(b)
        self.conv_b_3_bn = nn.BatchNorm2d(b)
        self.conv_c_1_bn = nn.BatchNorm2d(c_1)
        self.conv_c_2_bn = nn.BatchNorm2d(c)
        self.conv_c_3_bn = nn.BatchNorm2d(c)
        self.conv_c_4_bn = nn.BatchNorm2d(c)
        self.conv_d_bn = nn.BatchNorm2d(d)
        self.output = a + 2 * b + 2 * c + d

    def forward(self, x):
        x_a = F.relu(self.conv_a_bn(self.conv_a(x)))
        x_b = F.relu(self.conv_b_1_bn(self.conv_b_1(x)))
        x_b_1 = F.relu(self.conv_b_2_bn(self.conv_b_2(x_b)))
        x_b_2 = F.relu(self.conv_b_3_bn(self.conv_b_3(x_b)))
        x_c = F.relu(self.conv_c_1_bn(self.conv_c_1(x)))
        x_c = F.relu(self.conv_c_2_bn(self.conv_c_2(x_c)))
        x_c_1 = F.relu(self.conv_c_3_bn(self.conv_c_3(x_c)))
        x_c_2 = F.relu(self.conv_c_4_bn(self.conv_c_4(x_c)))
        x_d = F.max_pool2d(x, 3, stride=1, padding=1)
        x_d = F.relu(self.conv_d_bn(self.conv_d(x_d)))
        x = torch.cat([x_a, x_b_1, x_b_2, x_c_1, x_c_2, x_d], dim=1)
        assert x.shape[1] == self.output
        return x


class c2d_googlenet_v3(nn.Module):
    def __init__(self):
        super(c2d_googlenet_v3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0)  # 299 -> 149
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)  # 149 -> 147
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 147 -> 147 -> pool -> 73

        self.conv4 = nn.Conv2d(64, 80, kernel_size=3, stride=1, padding=0)  # 73 -> 71
        self.conv5 = nn.Conv2d(80, 192, kernel_size=3, stride=2, padding=0)  # 71 -> 35
        self.conv6 = nn.Conv2d(192, 288, kernel_size=3, stride=1, padding=1)  # 35 -> 35

        self.in1_1 = inception(288, 64, 96, 128, 32, 64, 32)
        self.in1_2 = inception(288, 64, 96, 128, 32, 64, 32)
        self.in1_3 = inception(288, 192, 128, 192, 128, 192, 192)

        self.in2_1 = inception_v3(768, 192, 128, 192, 128, 192, 192)
        self.in2_2 = inception_v3(768, 192, 128, 192, 128, 192, 192)
        self.in2_3 = inception_v3(768, 192, 128, 192, 128, 192, 192)
        self.in2_4 = inception_v3(768, 192, 128, 192, 128, 192, 192)
        self.in2_5 = inception_v3(768, 256, 128, 640, 128, 256, 128)

        self.in3_1 = inception_v3_exp(1280, 256, 160, 320, 32, 128, 128)
        self.in3_2 = inception_v3_exp(1280, 256, 160, 640, 32, 192, 128)

        self.fc = nn.Linear(2048, 50)

        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4_bn = nn.BatchNorm2d(80)
        self.conv5_bn = nn.BatchNorm2d(192)
        self.conv6_bn = nn.BatchNorm2d(288)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        # print(x.shape)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        # print(x.shape)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        # print(x.shape)
        x = F.max_pool2d(x, 3, stride=2)
        # print(x.shape)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        # print(x.shape)
        x = F.relu(self.conv5_bn(self.conv5(x)))
        # print(x.shape)
        x = F.relu(self.conv6_bn(self.conv6(x)))
        # print(x.shape)
        x = self.in1_1(x)
        x = self.in1_2(x)
        x = self.in1_3(x)
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        # print(x.shape)
        x = self.in2_1(x)
        x = self.in2_2(x)
        x = self.in2_3(x)
        x = self.in2_4(x)
        x = self.in2_5(x)
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        # print(x.shape)
        x = self.in3_1(x)
        x = self.in3_2(x)
        # print(x.shape)
        x = F.avg_pool2d(x, 8)
        x = x.view(-1, 2048)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        # print(x.shape)
        return F.log_softmax(x, dim=1)


class res_block(nn.Module):
    def __init__(self, input, output, stride=1):
        super(res_block, self).__init__()
        self.conv_1 = nn.Conv2d(input, output, kernel_size=3, stride=stride, padding=1)
        self.conv_2 = nn.Conv2d(output, output, kernel_size=3, stride=1, padding=1)
        self.conv_1_bn = nn.BatchNorm2d(output)
        self.conv_2_bn = nn.BatchNorm2d(output)
        if stride == 2:
            self.conv_tmp = nn.Conv2d(input, output, kernel_size=1, stride=2, padding=0)
            self.conv_tmp_bn = nn.BatchNorm2d(output)

    def forward(self, x):
        x_1 = F.relu(self.conv_1_bn(self.conv_1(x)))
        x_2 = self.conv_2_bn(self.conv_2(x_1))
        if x.shape[1] != x_2.shape[1]:
            x = self.conv_tmp_bn(self.conv_tmp(x))
        assert x.shape[1] == x_2.shape[1]
        assert x.shape[2] == x_2.shape[2]
        assert x.shape[3] == x_2.shape[3]
        x = F.relu(x_2 + x)
        return x


class resnet_18(nn.Module):
    def __init__(self):
        super(resnet_18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2_1 = res_block(64, 64)
        self.conv2_2 = res_block(64, 64)
        self.conv3_1 = res_block(64, 128, 2)
        self.conv3_2 = res_block(128, 128)
        self.conv4_1 = res_block(128, 256, 2)
        self.conv4_2 = res_block(256, 256)
        self.conv5_1 = res_block(256, 512, 2)
        self.conv5_2 = res_block(512, 512)
        self.fc = nn.Linear(512, 50)
        self.conv1_bn = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        # print(x.shape)
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        # print(x.shape)
        x = self.conv2_2(self.conv2_1(x))
        # print(x.shape)
        x = self.conv3_2(self.conv3_1(x))
        # print(x.shape)
        x = self.conv4_2(self.conv4_1(x))
        # print(x.shape)
        x = self.conv5_2(self.conv5_1(x))
        # print(x.shape)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 512)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        # print(x.shape)
        return F.log_softmax(x, dim=1)
