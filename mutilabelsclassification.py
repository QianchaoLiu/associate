# encoding=utf-8
# muti-labels classification for Xing at 2016/09/09
# with libsvm to classification type of exercise
import jieba
import os
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
                            if i>0:
                                line_split = line.split('\t')
                                pair.append([line_split[3].strip(' '),line_split[4]])
        print len(pair),"dataset has been loaded,", "the number of types:",str(self._typecount(pair))
        return pair


if __name__ == "__main__":
    svm = svmdemo()
    for text_type in svm.readdat('/Users/liuqianchao/Desktop/dat'):

        text = text_type[0]
        type = text_type[1]
        seg_list = jieba.cut(text, cut_all=False)
        print ' '.join(seg_list)
        print type
        break