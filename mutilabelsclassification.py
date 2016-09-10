# encoding=utf-8
# muti-labels classification for Xing at 2016/09/09
# with svm to classification type of exercise
import jieba
import jieba.analyse
import os
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

import sys
reload(sys)
sys.setdefaultencoding('utf8')

class svmdemo(object):

    def __init__(self):
        pass

    def _typecount(self,dataset):
        return len(set([text_type[1] for text_type in dataset]))

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
                                pair.append([line_split[3].strip(' ').encode('utf-8'), line_split[4]])
                                print line_split[4]
        print len(pair),"dataset has been loaded,", "the number of types:",str(self._typecount(pair))
        return pair


if __name__ == "__main__":
    svm = svmdemo()
    seg_document = []
    for text_type in svm.readdat('/Users/liuqianchao/Desktop/dat'):

        text, _type = text_type[0], text_type[1]
        seg_list = list(jieba.cut(text, cut_all=False))
        seg_document.append(' '.join(seg_list))

        # 使用textrank提取关键词
        # for item in jieba.analyse.textrank(text):
        #     print item

    tfidf_vectorizer = TfidfVectorizer()
    real_vec = tfidf_vectorizer.fit_transform(seg_document)
    seg_vec = real_vec.toarray()
    print len(seg_vec)
    print seg_vec[0]
