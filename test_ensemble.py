import matplotlib
matplotlib.use('Agg')
import torch
from torch.autograd import Variable
from torchvision import transforms
from model.cnn import *
import os
from PIL import Image

path_1 = 'save/1-frame-c2d-res18-1/net-epoch-20.pkl'
path_2 = 'save/1-frame-c2d-res18-2/net-epoch-20.pkl'
path_3 = 'save/1-frame-c2d-res18-3/net-epoch-20.pkl'
path_4 = 'save/1-frame-c2d-res18-4/net-epoch-14.pkl'
model_1 = c2d_resnet_18().cuda()
model_1.load_state_dict(torch.load(path_1))
model_1.eval()
model_2 = c2d_resnet_18().cuda()
model_2.load_state_dict(torch.load(path_2))
model_2.eval()
model_3 = c2d_resnet_18().cuda()
model_3.load_state_dict(torch.load(path_3))
model_3.eval()
model_4 = c2d_resnet_18().cuda()
model_4.load_state_dict(torch.load(path_4))
model_4.eval()

transform = transforms.Compose([transforms.Scale(256),
                                transforms.RandomCrop(227),
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

        # 10-crop
        data_1 = transform(Image.open(current_path + fn2)).view(1, 3, 227, 227)
        data_2 = transform(Image.open(current_path + fn2)).view(1, 3, 227, 227)
        data_3 = transform(Image.open(current_path + fn2)).view(1, 3, 227, 227)
        data_4 = transform(Image.open(current_path + fn2)).view(1, 3, 227, 227)
        data_5 = transform(Image.open(current_path + fn2)).view(1, 3, 227, 227)
        data_6 = transform(Image.open(current_path + fn2)).view(1, 3, 227, 227)
        data_7 = transform(Image.open(current_path + fn2)).view(1, 3, 227, 227)
        data_8 = transform(Image.open(current_path + fn2)).view(1, 3, 227, 227)
        data_9 = transform(Image.open(current_path + fn2)).view(1, 3, 227, 227)
        data_10 = transform(Image.open(current_path + fn2)).view(1, 3, 227, 227)
        data = torch.cat([data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10], dim=0)
        data = Variable(data.cuda())

        # 4-resnet-ensemble
        crop_voting_list = [0] * 50
        output = model_1(data)
        pred_clip_1 = (output.data.max(1, keepdim=True)[1]).cpu().numpy()
        for i in pred_clip_1:
            crop_voting_list[int(i)] += 1
        pred_clip_1 = crop_voting_list.index(max(crop_voting_list))

        crop_voting_list = [0] * 50
        output = model_2(data)
        pred_clip_2 = (output.data.max(1, keepdim=True)[1]).cpu().numpy()
        for i in pred_clip_2:
            crop_voting_list[int(i)] += 1
        pred_clip_2 = crop_voting_list.index(max(crop_voting_list))

        crop_voting_list = [0] * 50
        output = model_3(data)
        pred_clip_3 = (output.data.max(1, keepdim=True)[1]).cpu().numpy()
        for i in pred_clip_3:
            crop_voting_list[int(i)] += 1
        pred_clip_3 = crop_voting_list.index(max(crop_voting_list))

        crop_voting_list = [0] * 50
        output = model_4(data)
        pred_clip_4 = (output.data.max(1, keepdim=True)[1]).cpu().numpy()
        for i in pred_clip_4:
            crop_voting_list[int(i)] += 1
        pred_clip_4 = crop_voting_list.index(max(crop_voting_list))

        ensemble_voting_list = [0] * 50
        ensemble_voting_list[pred_clip_1] += 1
        ensemble_voting_list[pred_clip_2] += 1
        ensemble_voting_list[pred_clip_3] += 1
        ensemble_voting_list[pred_clip_4] += 1
        pred_clip = ensemble_voting_list.index(max(ensemble_voting_list))
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