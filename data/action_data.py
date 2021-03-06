# coding: utf-8
import cv2
from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils import data

# load dataset
class ucf_50(data.Dataset):
    def __init__(self, flag=0, scale=256, size=224, length=6, image_ready=0):  # 0-train, 1-val
        self.length = length
        if image_ready:
            path = 'data/train/' if not flag else 'data/val/'
            label_cnt = 0
            self.imgs = []
            self.labels = []
            for fn1 in os.listdir(path):
                for fn2 in os.listdir(path + fn1):
                    self.imgs += [path + fn1 + '/' + fn2]
                    self.labels += [label_cnt]
                label_cnt += 1
            assert len(self.imgs) == len(self.labels)
        else:
            self.imgs, self.labels = self.extract_clips(length, flag)
        self.transform = transforms.Compose([transforms.Scale(scale),
                                             transforms.RandomCrop(size),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        # return the number of training or testing samples
        return len(self.imgs)

    def __getitem__(self, idx):
        # return the tuple (img, label) by the index
        # c2d
        img = self.transform(Image.open(self.imgs[idx]))
        label = self.labels[idx]
        # # c3d
        # img = self.transform(Image.open(self.imgs[idx * self.length]))
        # for i in range(self.length - 1):
        #     img_tmp = self.transform(Image.open(self.imgs[idx * self.length + i + 1]))
        #     img = torch.cat([img, img_tmp], dim=0)
        # label = self.labels[idx * self.length]
        return img, label

    def extract_clips(self, extract_length, flag=0):
        imgs = []
        labels = []
        data_folder = os.getcwd() + '/data'
        result_path = data_folder + '/train/' if not flag else data_folder + '/val/'
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        path = data_folder + '/../download/UCF50/'
        label_cnt = 0
        for fn1 in os.listdir(path):
            current_path = path + fn1 + '/'
            if not os.path.exists(result_path + fn1):
                os.mkdir(result_path + fn1)
            for fn2 in os.listdir(current_path):
                if '.avi' in fn2 and ((not flag and (int(fn2[-10]) * 10 + int(fn2[-9]) > 1))
                or (flag and (int(fn2[-10]) * 10 + int(fn2[-9]) <= 1))):
                    cap = cv2.VideoCapture(current_path + fn2)
                    # frames_width = cap.get(3)
                    # frames_height = cap.get(4)
                    num_frames = cap.get(7)
                    # print(frames_width, frames_height, num_frames)
                    frames_cnt = 0
                    read = 0
                    while (cap.isOpened()):
                        # 第一个参数ret的值为True或False，代表有没有读到图片，第二个参数是frame，是当前截取一帧的图片。
                        ret, frame = cap.read()
                        if not read and frames_cnt == num_frames // 2:
                            read = 1
                            frames_cnt = 0
                        if read:
                            cv2.imwrite(result_path + fn1 + '/' + fn2[:-4] + '_' + str(frames_cnt) + '.jpg', frame)
                            imgs += [result_path + fn1 + '/' + fn2[:-4] + '_' + str(frames_cnt) + '.jpg']
                            labels += [label_cnt]
                            if frames_cnt == extract_length - 1:
                                break
                        frames_cnt += 1
                    cap.release()
                    print('processing ' + fn2)
                # input()
            label_cnt += 1
        assert len(imgs) == len(labels)
        return imgs, labels
