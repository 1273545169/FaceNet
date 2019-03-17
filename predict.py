# 此文件用于预测

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from model import vgg16
import tensorflow as tf
import numpy as np
import argparse
import functions
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # mpimg 用于读取图片


def main(args):
    with tf.Session() as sess:

        inputs_codes = tf.placeholder(tf.float32, shape=[None, args.codes_size], name="inputs")
        embeddings = functions.model_add(inputs_codes, args.embedding_size)

        emb_array, label_array = get_array(args)

        i = int(input("请选择：(1存储，2预测)"))
        if i == 1:

            x = int(input("请选择：(1批量存储，2手动添加)"))
            if x == 1:
                # img保存图片
                img = []
                # emb_array图片编码、label_array图片标签
                emb_array = None
                label_array = []

                files = os.listdir(args.data_path)

                for ii, filename in enumerate(files, 1):

                    image_path = os.path.join(args.data_path, filename)
                    img.append(functions.load_image(image_path))

                    # 保存图片的标签
                    label_array.append(os.path.basename(image_path).split('.')[0])

                    if ii % args.batch_size == 0 or ii == len(files):

                        emb = get_embeddings(args, sess, inputs_codes, embeddings, np.array(img))
                        img = []

                        # 保存embeddding
                        if emb_array is None:
                            emb_array = emb
                        else:
                            emb_array = np.concatenate((emb_array, emb))

                # 将label和embeddding存入文件中
                set_array(args, emb_array, label_array)

            elif x == 2:

                chose = 1
                while chose == 1:
                    filename = input("请输入图片名称：")
                    image_path = os.path.join(args.data_path, filename + '.jpg')
                    emb = get_embeddings(args, sess, inputs_codes, embeddings, functions.load_image(image_path))

                    # 保存label
                    if label_array is None:
                        label_array = np.array(os.path.basename(image_path).split('.')[0])
                    else:
                        label_array = np.append(label_array, os.path.basename(image_path).split('.')[0])

                    # 保存embeddding
                    if emb_array is None:
                        emb_array = emb
                    else:
                        emb_array = np.concatenate((emb_array, emb))
                    # 将label和embeddding存入文件中
                    set_array(args, emb_array, label_array)

                    chose = int(input("是否继续输入：(1继续，2结束)"))

            else:
                print("错误输入")

        elif i == 2:

            filename = input("请输入图片名称：")
            start = time.time()
            image_path = os.path.join(args.predict_data_path, filename + '.jpg')
            img = functions.load_image(image_path)
            emb = get_embeddings(args, sess, inputs_codes, embeddings, img)

            # emb = get_embeddings(args, sess, inputs_codes, embeddings, image_path)
            dist = np.sum(np.square(emb - emb_array), 1)

            index = np.argsort(dist)[0:5]
            label = list(label_array[index])
            name = max(set(label), key=label.count)
            # print(time.time() - start)
            print(name)

        else:
            print('错误输入')


# array预先存入的图片的编码和对应的标签
def get_array(args):
    emb_array = None
    label_array = None
    # 文件存在且不为空
    if os.path.exists(args.emb_array_path) and os.path.getsize(args.emb_array_path):
        emb_array = np.load(args.emb_array_path)
        label_array = np.load(args.label_array_path)
    return emb_array, label_array


def set_array(args, emb_array, label_array):
    np.save(args.emb_array_path, emb_array)
    np.save(args.label_array_path, label_array)


def get_embeddings(args, sess, inputs_codes, embeddings, img):
    # 计算codes and labels========================
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
    # 构建VGG16模型对象
    vgg = vgg16.Vgg16()
    vgg.build(input_)
    # 计算特征值
    test_codes = sess.run(vgg.flatten, feed_dict={input_: img.reshape((-1, 224, 224, 3))})

    # 计算embeddings=============================
    saver = tf.train.Saver()
    saver.restore(sess, args.model_path)

    emb = sess.run(embeddings, feed_dict={inputs_codes: test_codes})

    # return emb , label
    return emb


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str,
                        help='Directory where to write trained models and checkpoints.', default='model/checkpoints/')
    # 预先存储的图片路径
    parser.add_argument('--data_path', type=str,
                        help='Path to the data directory containing predict_set.',
                        default='predict_set/1/')
    # 用于预测的图片路径
    parser.add_argument('--predict_data_path', type=str,
                        help='Path to the data directory containing predict_set.',
                        default='predict_set/2/')

    parser.add_argument('--batch_size', type=str,
                        help='Number of images in a batch.',
                        default=20)

    parser.add_argument('--emb_array_path', type=str,
                        help='', default='predict_set/emb_array.npy')
    parser.add_argument('--label_array_path', type=str,
                        help='', default='predict_set/label_array.npy')
    parser.add_argument('--codes_size', type=int,
                        help='Dimensionality of the codes.', default=25088)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
