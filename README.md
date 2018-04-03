action_recognition
==================
The repository is a simple example for human action (in video) recognition. 

Pre-requisites
--------------
pytorch 0.3.1

torchvision 0.1.9

opencv-contrib-python 3.4.0.12

If there exists errors in your projects, please verify the package version since it's such annoying!

Datasets
--------
I now only test it on [UCF50](http://crcv.ucf.edu/data/UCF50.php). Maybe test it on UCF101 and HMDB51 in future!

The first time you run train_and_test.py, image_ready should be set to 0 to extract frames from the videos.

Model
-----
I now only support single-frame-training like alexnet. Maybe support multi-frame-training and two-stream-training in future!

Validation
----------
I use 'Leave-one-group-out-cross-validation' following the suggestions by UCF50. For robustness, I test 10 crops with the sampled frame for the optim result. Here shows clip@hit and video@hit results. Details can be found in test.py.

Date            Model            Clip@Hit(%)            video@Hit(%)

20180401        alexnet          53.84                  58.00

20180401        vgg16            65.86                  72.00

20180402        googlenet        59.67                  64.00

20180402        googlenet_v3     60.59                  66.00

20180402        res18            61.90                  66.00

20180403        4_res18s         62.21                  70.00 
