# encoding=utf-8
# muti-labels classification for Xing at 2016/09/09
# with softmax regression(implement by Tensorflow) to classification type of exercise
import jieba
import jieba.analyse
import os
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
reload(sys)
sys.setdefaultencoding('utf8')

class svmdemo(object):

    def __init__(self):
        pass

    @staticmethod
    def _typecount(dataset):
        _type_list = []
        for text_type in dataset:
            for _type in text_type[1]:
                _type_list.append(_type)
        return len(set(_type_list))

    def readdat(self,path):
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

    @staticmethod
    def softmax_regression(input_item_length=84140, output_item_length=246):
        x = tf.placeholder(tf.float32, [None, input_item_length]) # None means that a dimension can be of any length
        W = tf.Variable(tf.zeros([input_item_length, output_item_length]))
        b = tf.Variable(tf.zeros[output_item_length])

        pass


if __name__ == "__main__":
    svm = svmdemo()
    seg_document = []
    for text_type in svm.readdat('/Users/liuqianchao/Desktop/dat'):

        text, _type = text_type[0], text_type[1]
        seg_list = list(jieba.cut(text, cut_all=False))
        seg_document.append(' '.join(seg_list))
        # print text,str(_type).decode("string_escape")
        # 使用textrank提取关键词
        # for item in jieba.analyse.textrank(text):
        #     print item

    tfidf_vectorizer = TfidfVectorizer()
    real_vec = tfidf_vectorizer.fit_transform(seg_document)
    seg_vec = real_vec.toarray()
    print len(seg_vec)
    # mnist: 55000 data 28*28=784 dimensions of vector, and the output is 55000 * 10 dimension
    # in this example, 407481 data, each data with dimension of 84140, and the output is 407481 * 246(number of knowledge)
    print len(seg_vec[0])
