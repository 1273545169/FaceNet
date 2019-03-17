# 此文件用于计算测试集的codes和labels
import argparse
import sys
import os
import tensorflow as tf
import functions
import numpy as np
from model import vgg16

def main(args):

    # 用codes来存储特征值，用labels来存储羊的label
    codes , labels = get_codes(args)

    # batch数组用来临时存储图片数据
    batch = []

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

                # 保存label
                if labels is None:
                    labels = np.array(each)
                else:
                    labels = np.append(labels, each)

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
    np.save(args.codes_path, codes)
    np.save(args.labels_path, labels)


def get_codes(args):
    codes = None
    labels = None
    # 文件存在且不为空
    if os.path.exists(args.codes_path) and os.path.getsize(args.codes_path):
        codes = np.load(args.codes_path)
        labels = np.load(args.labels_path)
    return codes , labels


def parse_arguments(argv):
    parser = argparse.ArgumentParser()


    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing  face patches.',
                        default='G:/羊头照片整理/测试集/')

    parser.add_argument('--codes_path', type=str,
                        help='Path to the data directory containing val_codes.',
                        default='data/val/codes.npy')
    parser.add_argument('--labels_path', type=str,
                        help='Path to the data directory containing val_labels.',
                        default='data/val/labels.npy')

    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=10)

    parser.add_argument('--codes_size', type=int,
                        help='Dimensionality of the codes.', default=25088)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))




