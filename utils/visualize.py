from matplotlib import pyplot as plt
import numpy as np
import time


def draw_curves(epoch, train_loss, test_loss, train_accuracy, test_accuracy,
                save_name=time.strftime('%Y%m%d%H%M%S', time.localtime())):
    plt.close()  # close the former figure and update it
    plt.subplot(121)
    plt.plot(epoch, train_loss, 'b')
    plt.plot(epoch, test_loss, 'r')
    plt.title('loss')
    plt.subplot(122)
    plt.plot(epoch, train_accuracy, 'b')
    plt.plot(epoch, test_accuracy, 'r')
    plt.title('accuracy')
    plt.savefig('save/' + save_name)
    plt.show()
    plt.pause(1)


def draw_feature(data, save_name=time.strftime('%Y%m%d%H%M%S', time.localtime()), padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    print(padding)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    print(data.shape)
    plt.imshow(data[:, :, 0:3])
    plt.title('conv1')
    plt.axis('off')
    plt.savefig('save/' + save_name + '-conv1')
    plt.show()
    plt.pause(1)