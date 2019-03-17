# 此文件用于计算训练集的codes和labels，并将其存到硬盘上
import argparse
import sys
import os
import tensorflow as tf
import time
from sklearn.model_selection import StratifiedShuffleSplit
import functions
import numpy as np
from model import vgg16

def main(args):

    # 用codes来存储特征值
    codes = None
    # 用labels来存储羊的label
    labels = []
    # 用images_path来存储照片的路径
    image_paths = []
    # batch数组用来临时存储图片数据
    batch = []
    # # num_per_class表示每一只羊图片张数
    # num_per_class = []

    contents = os.listdir(args.data_dir)
    classes = [each for each in contents if os.path.isdir(args.data_dir + each)]

    with tf.Session() as sess:
        # 构建VGG16模型对象
        vgg = vgg16.Vgg16()
        input_ = tf.placeholder(tf.float32, [None, 224, 224, 3] , name="input_images")
        with tf.name_scope("content_cnn"):
            # 载入VGG16模型
            vgg.build(input_)

        # 对每只羊分别用VGG16计算特征值
        for each in classes:
            print("Starting {} images".format(each))
            class_path = args.data_dir + each
            files = os.listdir(class_path)

            for ii, file in enumerate(files, 1):

                labels.append(each)
                image_paths.append(os.path.join(class_path, file))

                # 载入图片并放入batch数组中
                img = functions.load_image(os.path.join(class_path, file))
                batch.append(img.reshape((1, 224, 224, 3)))

                # 如果图片数量到了batch_size则开始具体的运算
                if ii % args.batch_size == 0 or ii == len(files):

                    images = np.concatenate(batch)

                    feed_dict = {input_: images}
                    # 计算特征值
                    codes_batch = sess.run(vgg.flatten, feed_dict=feed_dict)

                    # 将结果放入到codes数组中
                    if codes is None:
                        codes = codes_batch

                    else:
                        codes = np.concatenate((codes, codes_batch))

                    # 清空数组准备下一个batch的计算
                    batch = []
                    print('{} images processed'.format(ii))
                    # num_per_class.append(ii)

    # 得到一个 codes 数组，一个 labels 数组，一个image_paths数组，分别存储了所有羊的特征值和类别以及路径,然后将数据再保存到硬盘上
    # codes.tofile('data/codes')
    np.save('data/train/codes.npy', codes)

    # with open('data/labels', 'w') as f:
    #     writer = csv.writer(f, delimiter='\n')
    #     writer.writerow(labels)

    np.save('data/train/labels.npy', labels)


def get_codes():

    # labels=[]

    # codes = np.fromfile('data/codes', dtype=np.float32).reshape(-1, codes_size)
    codes=np.load('data/train/codes.npy')

    # with open('data/labels') as f:
    #     reader = csv.reader(f)
    #     for i in reader:
    #         labels.append(i)
    #     labels = list(itertools.chain.from_iterable(labels))
    # print(labels)

    labels=np.load('data/train/labels.npy')

    # # 使用 StratifiedShuffleSplit 方法来把数据进行分层随机划分，训练集：测试集 = 8:2
    # ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    #
    # train_idx, val_idx = next(ss.split(codes, labels))
    #
    # train_x, train_y = codes[train_idx], np.array(labels)[train_idx]
    # val_x, val_y = codes[val_idx], np.array(labels)[val_idx]

    return codes , labels , codes.shape[1]

    # return train_x, train_y, val_x, val_y, train_x.shape[1]



def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing  train_set.',
                        default='G:/羊头照片整理/训练集/')

    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=20)

    parser.add_argument('--codes_size', type=int,
                        help='Dimensionality of the codes.', default=25088)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))



