# -*- coding:utf-8 -*-

""" 加载数据的代码. """

import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import get_images

FLAGS = flags.FLAGS

class DataGenerator(object):
    """
    数据生成器负责产生一个批次的sinusoid函数或者Omniglot数据. 一个 "class" 相当于omniglot的一个字母或者一个特定振幅、相位的正弦函数.
    """
    def __init__(self, num_samples_per_class, batch_size, config={}):
        """
        参数:
            num_samples_per_class: 每个批次下每个类下的样本数
                                   1. 正弦拟合任务为每个批次每个特定正弦函数下的采样个数
                                   2. 图像分类任务为每个字符（或图片类别）下的采样图片个数
            batch_size: meta批次的大小 (正弦任务为不同振幅、相位的函数个数)
        """
        "批次大小"
        self.batch_size = batch_size
        "每个类别(函数)下的采样个数"
        self.num_samples_per_class = num_samples_per_class
        "self.num_classes默认值为1，仅适用于分类任务"
        self.num_classes = 1

        if FLAGS.datasource == 'sinusoid':
            self.generate = self.generate_sinusoid_batch
            "振幅的随机采样范围"
            self.amp_range = config.get('amp_range', [0.1, 5.0])
            "相位的随机采样范围"
            self.phase_range = config.get('phase_range', [0, np.pi])
            "数据采样点的输入范围"
            self.input_range = config.get('input_range', [-5.0, 5.0])
            "sinusoid的输入维度"
            self.dim_input = 1
            "sinusoid的输出维度"
            self.dim_output = 1
        elif 'omniglot' in FLAGS.datasource:
            "self.num_classes==FLAGS.num_classes"
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            "omniglot图片的输入尺寸"
            self.img_size = config.get('img_size', (28, 28))
            "omniglot任务对应的输入维度，28x28, (channels=1)"
            self.dim_input = np.prod(self.img_size)
            "omniglot任务的输出维度，few-shot对应的类别数"
            self.dim_output = self.num_classes
            # data that is pre-resized using PIL with lanczos filter
            "数据文件夹的根路径"
            data_folder = config.get('data_folder', './data/omniglot_resized')
            "character_folders=omniglot下每个字母文件夹的路径列表"
            character_folders = [os.path.join(data_folder, family, character) \
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
            "设置字母文件夹路径列表随机打乱的随机种子"
            random.seed(1)
            "打乱字母文件夹路径列表"
            random.shuffle(character_folders)
            "omniglot验证集(用于验证的字母(类)，集合中的元素不是样本而是字母，每个字母下才是样本)，默认为100"
            num_val = 100
            "omniglot训练集(用于训练的字母(类)，集合中的元素不是样本而是字母，每个字母下才是样本)，默认为1200-100"
            num_train = config.get('num_train', 1200) - num_val
            # self.metatrain_character_folders
            #######################################################################
            "用于meta train的字母(类)"
            self.metatrain_character_folders = character_folders[:num_train]
            # self.metaval_character_folders
            #######################################################################
            if FLAGS.test_set:
                "如果FLAGS.test_set为True, "
                "则没有验证类别集合，只有测试类别集合"
                self.metaval_character_folders = character_folders[num_train+num_val:]
            else:
                "如果FLAGS.test_set为False, "
                "则有验证类别集合，验证类别集是从指定的训练集中划出来的一部分"
                self.metaval_character_folders = character_folders[num_train:num_train+num_val]
            #######################################################################
            "数据增强，对数据进行旋转，旋转角度为0，90，180，270"
            self.rotations = config.get('rotations', [0, 90, 180, 270])
        elif FLAGS.datasource == 'miniimagenet':
            "miniImagenet每个few-shot的类别数"
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            "miniimagenet的图片尺寸"
            self.img_size = config.get('img_size', (84, 84))
            "miniimagenet的输入维度为: 84x84x3"
            self.dim_input = np.prod(self.img_size)*3
            "miniimagenet的输出维度为: 每个few-shot的类别数"
            self.dim_output = self.num_classes
            "miniimagenet的训练类别文件夹路径"
            metatrain_folder = config.get('metatrain_folder', './data/miniImagenet/images/train')
            if FLAGS.test_set:
                "如果为测试，则metaval_folder为测试文件夹路径"
                metaval_folder = config.get('metaval_folder', './data/miniImagenet/images/test')
            else:
                "如果不为测试，则metaval_folder为验证文件夹路径"
                metaval_folder = config.get('metaval_folder', './data/miniImagenet/images/val')
            "获取训练文件夹下的类别文件夹路径"
            self.metatrain_character_folders = [os.path.join(metatrain_folder, label) \
                for label in os.listdir(metatrain_folder) \
                if os.path.isdir(os.path.join(metatrain_folder, label)) \
                ]
            "获取验证/测试文件夹下的类别文件夹路径"
            self.metaval_character_folders = [os.path.join(metaval_folder, label) \
                for label in os.listdir(metaval_folder) \
                if os.path.isdir(os.path.join(metaval_folder, label)) \
                ]
            "不做旋转数增强"
            self.rotations = config.get('rotations', [0])
        else:
            "如果超出了：正弦拟合/omniglot/miniimagenet, 则抛出错误"
            raise ValueError('不可识别的数据源')


    def make_data_tensor(self, train=True):
        "该函数在omniglot任务和miniimagenet任务下调用，产生数据构造对应的计算图"
        if train:
            "如果是训练阶段，"
            folders = self.metatrain_character_folders
            # number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
            num_total_batches = 200000
        else:
            folders = self.metaval_character_folders
            num_total_batches = 600

        # make list of files
        print('Generating filenames')
        all_filenames = []
        for _ in range(num_total_batches):
            sampled_character_folders = random.sample(folders, self.num_classes)
            random.shuffle(sampled_character_folders)
            labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)
            # make sure the above isn't randomized order
            labels = [li[0] for li in labels_and_images]
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)

        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        if FLAGS.datasource == 'miniimagenet':
            "如果是miniimagenet的数据集"
            "读取jpeg图片(uint8 tensor [84, 84, 3])"
            image = tf.image.decode_jpeg(image_file, channels=3)
            "调整jpeg的形状, [84，84，3]"
            image.set_shape((self.img_size[0],self.img_size[1],3))
            "调整为[84x84x3]的向量"
            image = tf.reshape(image, [self.dim_input])
            "类型转换，归一化"
            image = tf.cast(image, tf.float32) / 255.0
        else:
            "如果是omniglot的数据集"
            "读取png图片(uint8 tensor [24, 24, 1])"
            image = tf.image.decode_png(image_file)
            "调整png的形状, [24, 24, 1]"
            image.set_shape((self.img_size[0],self.img_size[1],1))
            "调整png数据的"
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
            image = 1.0 - image  # invert
        num_preprocess_threads = 1 # TODO - enable this to be set to >1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_class
        batch_image_size = self.batch_size  * examples_per_batch
        print('Batching images')
        images = tf.train.batch(
                [image],
                batch_size = batch_image_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_image_size,
                )
        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in range(self.batch_size):
            image_batch = images[i*examples_per_batch:(i+1)*examples_per_batch]

            if FLAGS.datasource == 'omniglot':
                # omniglot augments the dataset by rotating digits to create new classes
                # get rotation per class (e.g. 0,1,2,0,0 if there are 5 classes)
                rotations = tf.multinomial(tf.log([[1., 1.,1.,1.]]), self.num_classes)
            label_batch = tf.convert_to_tensor(labels)
            new_list, new_label_list = [], []
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(0, self.num_classes)
                class_idxs = tf.random_shuffle(class_idxs)

                true_idxs = class_idxs*self.num_samples_per_class + k
                new_list.append(tf.gather(image_batch,true_idxs))
                if FLAGS.datasource == 'omniglot': # and FLAGS.train:
                    new_list[-1] = tf.stack([tf.reshape(tf.image.rot90(
                        tf.reshape(new_list[-1][ind], [self.img_size[0],self.img_size[1],1]),
                        k=tf.cast(rotations[0,class_idxs[ind]], tf.int32)), (self.dim_input,))
                        for ind in range(self.num_classes)])
                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

    def generate_sinusoid_batch(self, train=True, input_idx=None):
        # Note train arg is not used (but it is used for omniglot method.
        # input_idx is used during qualitative testing --the number of examples used for the grad update
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
            if input_idx is not None:
                init_inputs[:,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_class-input_idx, retstep=False)
            outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])
        return init_inputs, outputs, amp, phase
