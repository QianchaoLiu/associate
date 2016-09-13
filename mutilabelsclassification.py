# encoding=utf-8
# muti-labels classification for Xing at 2016/09/09
# with softmax regression(implement by Tensorflow) to classification type of exercise
import jieba
import jieba.analyse
import os
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

import sys
reload(sys)
sys.setdefaultencoding('utf8')

class svmdemo(object):


    def __init__(self):
        self._index_in_epoch = 0
        self.num_examples = 0
        self._text = None
        self._labels = None

    def text(self,_text):
        self._text = _text

    def labels(self,_labels):
        self._labels = _labels

    @staticmethod
    def _typecount(dataset):
        _type_list = []
        for text_type in dataset:
            for _type in text_type[1]:
                _type_list.append(_type)
        return len(set(_type_list))

    def read_dat(self,path):
        pair = []
        if path == "":
            print 'empty data path'
        else:
            for root,dirs,files in os.walk(path):
                for file in files:
                    wd = os.path.join(root,file)
                    with open(wd) as rf:
                        for i,line in enumerate(rf):
                            # 剔除第一行乱码
                            if i>0:
                                line_split = line.split('\t')
                                # 文本,题目分类
                                content = line_split[3].strip(' ').encode('utf-8')
                                _type = line_split[4].split('-')
                                pair.append([content, _type])

        print len(pair),"dataset has been loaded,", "the number of types:",str(self._typecount(pair))
        return pair

    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_emamples:
            # TODO
            pass
        end = self._index_in_epoch
        return self._text[start:end], self._labels[start:end]

    def softmax_regression(self,input_item_length=84140, output_item_length=246):
        # define
        x = tf.placeholder(tf.float32, [None, input_item_length]) # None means that a dimension can be of any length
        W = tf.Variable(tf.zeros([input_item_length, output_item_length]))
        b = tf.Variable(tf.zeros[output_item_length])
        y = tf.nn.softmax(tf.matmul(x,W)+b)

        # train
        y_ = tf.placeholder(tf.float32, [None, output_item_length])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        for i in range(1000):
            batch_xs, batch_ys = self.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

if __name__ == "__main__":
    svm = svmdemo()
    seg_document = []
    labels_list = []

    for text_type in svm.read_dat('/Users/liuqianchao/Desktop/dat'):

        text, _type = text_type[0], text_type[1]
        seg_list = list(jieba.cut(text, cut_all=False))
        seg_document.append(' '.join(seg_list))

        #label_list.append(_type)
        labels_list.append(_type)
        # print text,str(_type).decode("string_escape")
        # 使用textrank提取关键词
        # for item in jieba.analyse.textrank(text):
        #     print item
    labels = reduce(lambda x, y: x+y, labels_list)


    labels = set(labels)
    print 'size of output',len(labels)
    labels_dict = {}
    for label in labels:
        labels_dict[label] = len(labels_dict)


    _label = []
    for item in labels_list:
        tmp = []
        for single_label in item:
            tmp.append(labels_dict[single_label])
        _label.append(tmp)

    tfidf_vectorizer = TfidfVectorizer()
    real_vec = tfidf_vectorizer.fit_transform(seg_document)
    # 输入
    seg_vec = real_vec.toarray()
    seg_vec = np.array(seg_vec)

    svm._text = seg_vec
    svm._labels = _label

    num_examples = len(seg_vec) # 407481
    svm.num_examples = num_examples


    # mnist: 55000 data 28*28=784 dimensions of vector, and the output is 55000 * 10 dimension
    # in this example, 407481 data, each data with dimension of 84140, and the output is 407481 * 246(number of knowledge)
    print len(seg_vec[0]) # 84140

    # implement of tensorflow
    svm.text()
    svm.labels()

