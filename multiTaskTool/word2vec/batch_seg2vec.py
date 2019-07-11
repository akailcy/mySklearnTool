from gensim.models import Word2Vec
import os
import time
import logging
from tqdm import tqdm
import jieba
import gc
WORK_PATH = 'model/'

class Word2VecTrainingMaster():

    def __init__(self, new_work=True, corpus_path='', work_path='', base_name='word2vec.model', num_features=128, min_word_count=5, context=4, auto=True, workers=12,batch_size=500000, step_save=False):
        self.work_path = work_path
        self.corpus_path = corpus_path
        self.word2vec_model = self.make_word2vec_model(num_features, min_word_count, context,workers)
        self.files=os.listdir(corpus_path)
        print("reader load down")
        self.base_name=base_name
        self.word2vec_model.save(self.work_path + '/' + self.base_name)
        self.step_save = step_save
        self.auto=auto
        self.batch_size=batch_size

    def train(self):
        print("training begin")
        logging.info("training begin")
        if self.auto:
            INIT = True
            for i,f in enumerate(self.files):
                start=time.time()
                with open(self.corpus_path+f,'r',encoding='utf-8') as m:
                    print("read {}".format(f))
                    lines=[line.strip() for line in m.readlines()]
                lines=list(map(lambda x:list(jieba.cut(x)),lines))
                end=time.time()
                print("take {} s".format(int(end-start)))
                for batch_index in tqdm(range(0,len(lines),self.batch_size)):
                    batch=lines[batch_index:batch_index+self.batch_size]
                    self.update_model(batch, init=INIT)
                    INIT = False
                self.word2vec_model.save('{}/{}_{}'.format(self.work_path ,i, self.base_name))
                del lines
                gc.collect()
        
    def make_word2vec_model(self, num_features, min_word_count, context,workers):
        return Word2Vec(size=num_features, min_count=min_word_count, window=context,workers=workers)

    def update_model(self, batch, init):
        if init:
            self.word2vec_model.build_vocab(batch)
        else:
            self.word2vec_model.build_vocab(batch, update=True)
        self.word2vec_model.train(batch, total_examples=self.word2vec_model.corpus_count, epochs=self.word2vec_model.iter)
        if self.step_save:
            self.word2vec_model.save(self.work_path + '/' + self.base_name)

if __name__ == '__main__':
    start=time.time()
    w2v = Word2VecTrainingMaster(corpus_path='med_segs/', num_features=128, work_path=WORK_PATH)
    w2v.train()
    end=time.time()
    print("take {} s".format(int(end-start)))

