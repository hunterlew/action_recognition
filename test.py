import torch
from torch.autograd import Variable
from torchvision import transforms
from model.cnn import c2d_alexnet
import os
import cv2
from PIL import Image

path = 'save/1-frame-c2d-alexnet/net-epoch-15.pkl'
model = c2d_alexnet().cuda()
model.load_state_dict(torch.load(path))
model.eval()
transform = transforms.Compose([transforms.Resize(227),
                                 transforms.CenterCrop(227),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])

video_path = 'download/UCF50/'
video_sum_cnt = 0
video_correct_cnt = 0
clip_sum_cnt = 0
clip_correct_cnt = 0
label_cnt = 0
for fn1 in os.listdir(video_path):
    current_path = video_path + fn1 + '/'
    for fn2 in os.listdir(current_path):
        if '.avi' in fn2 and (int(fn2[-10]) * 10 + int(fn2[-9]) <= 8):
            cap = cv2.VideoCapture(current_path + fn2)
            num_frames = cap.get(7)
            cnt = 0
            voting_list = [0] * 50
            while (cap.isOpened()):
                ret, frame = cap.read()
                # if ret and cnt < num_frames:
                if ret:
                    if (cnt == num_frames // 7) or (cnt == num_frames * 2 // 7) or (cnt == num_frames * 3 // 7) \
                    or (cnt == num_frames * 4 // 7) or (cnt == num_frames * 5 // 7) or (cnt == num_frames * 6 // 7):
                        clip_sum_cnt += 1
                        data = transform(Image.fromarray(frame.astype("uint8")))
                        data = data.view(1, 3, 227, 227)
                        data = Variable(data.cuda())
                        output = model(data)
                        pred_clip = int(output.data.max(1, keepdim=True)[1].cpu().numpy())
                        # print(pred_clip)
                        voting_list[pred_clip] += 1
                        if pred_clip == label_cnt:
                            clip_correct_cnt += 1
                else:
                    break
                cnt += 1
            cap.release()
            video_sum_cnt += 1
            print(voting_list)
            pred_video = voting_list.index(max(voting_list))
            if pred_video == label_cnt:
                video_correct_cnt += 1
            print('processing ' + fn2)

            print('\nClip hit accuracy: {}/{} ({:.2f}%)'.format(
                clip_correct_cnt, clip_sum_cnt, 100. * clip_correct_cnt / clip_sum_cnt))
            print('\nVideo hit accuracy: {}/{} ({:.2f}%)'.format(
                video_correct_cnt, video_sum_cnt, 100. * video_correct_cnt / video_sum_cnt))

    label_cnt += 1
