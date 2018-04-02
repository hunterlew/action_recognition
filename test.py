import matplotlib
matplotlib.use('Agg')
import torch
from torch.autograd import Variable
from torchvision import transforms
from model.cnn import *
import os
from PIL import Image

path = 'save/1-frame-c2d-res18/net-epoch-14.pkl'
model = c2d_resnet_18().cuda()
model.load_state_dict(torch.load(path))
model.eval()
transform = transforms.Compose([transforms.Scale(256),
                                transforms.RandomCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

video_path = 'data/val/'
video_sum_cnt = 0
video_correct_cnt = 0
clip_sum_cnt = 0
clip_correct_cnt = 0
label_cnt = 0
for fn1 in os.listdir(video_path):
    video_sum_cnt += 1
    voting_list = [0] * 50
    current_path = video_path + fn1 + '/'
    for fn2 in os.listdir(current_path):
        clip_sum_cnt += 1
        crop_voting_list = [0] * 50
        data_1 = transform(Image.open(current_path + fn2)).view(1, 3, 224, 224)
        data_2 = transform(Image.open(current_path + fn2)).view(1, 3, 224, 224)
        data_3 = transform(Image.open(current_path + fn2)).view(1, 3, 224, 224)
        data_4 = transform(Image.open(current_path + fn2)).view(1, 3, 224, 224)
        data_5 = transform(Image.open(current_path + fn2)).view(1, 3, 224, 224)
        data_6 = transform(Image.open(current_path + fn2)).view(1, 3, 224, 224)
        data_7 = transform(Image.open(current_path + fn2)).view(1, 3, 224, 224)
        data_8 = transform(Image.open(current_path + fn2)).view(1, 3, 224, 224)
        data_9 = transform(Image.open(current_path + fn2)).view(1, 3, 224, 224)
        data_10 = transform(Image.open(current_path + fn2)).view(1, 3, 224, 224)
        data = torch.cat([data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10], dim=0)
        data = Variable(data.cuda())
        output = model(data)
        pred_clip = (output.data.max(1, keepdim=True)[1]).cpu().numpy()
        for i in pred_clip:
            crop_voting_list[int(i)] += 1
        pred_clip = crop_voting_list.index(max(crop_voting_list))
        voting_list[pred_clip] += 1
        if pred_clip == label_cnt:
            clip_correct_cnt += 1
    pred_video = voting_list.index(max(voting_list))
    if pred_video == label_cnt:
        video_correct_cnt += 1
    print('processing ' + fn1)

    print('\nClip hit accuracy: {}/{} ({:.2f}%)'.format(
        clip_correct_cnt, clip_sum_cnt, 100. * clip_correct_cnt / clip_sum_cnt))
    print('\nVideo hit accuracy: {}/{} ({:.2f}%)'.format(
        video_correct_cnt, video_sum_cnt, 100. * video_correct_cnt / video_sum_cnt))

    label_cnt += 1
