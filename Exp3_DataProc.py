"""
这个文件中可以添加数据预处理相关函数和类等
如词汇表生成，Word转ID（即词转下标）等
此文件为非必要部分，可以将该文件中的内容拆分进其他部分
"""
import json
import os

from Exp3_Config import Training_Config
from torch.utils.data import DataLoader
from Exp3_Model import TextCNN_Model
import torch
import numpy as np
from torchtext import data
from torchtext.vocab import build_vocab_from_iterator
from torch.nn import init
from tqdm import tqdm
from collections import Counter
import jieba
from Exp3_Config import Training_Config
import torch.nn as nn
from Exp3_Config import Training_Config
config = Training_Config()

class SentenceProcess:
    def __init__(self):
        pass

    def tokenizer(self,sentence):
        return sentence

    def build_vocab(self,filepath,vocabsize):
        '''

        :param filepath: filepath of train.txt
        :param vocabsize: size of vocab
        :build_vocab_from_iterator([counter_sorted]) toechtext.vocab
        :return: None
        '''
        counter = Counter()
        with open(filepath, encoding='utf-8') as f:
            for line_ in f:
                counter.update(self.tokenizer(line_.strip()))
        counter_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        counter_sorted = dict(counter_sorted[:vocabsize])

        vocab = build_vocab_from_iterator([counter_sorted])
        # print(vocab.get_stoi())

        dict_ = vocab.get_stoi()
        # print('len dict', len(dict_))
        assert type(dict_) != 'dict', 'not dict!'
        path = config.path_root+config.vocab_path
        np.save(path, dict_)
        print('Successfully build vocab!\n Save as Dictionary in {}'.format(path))

    def sentence2index(self, dictionary, sentence):
        # let look up in dictionary and convert words to number!
        sentence_id = []
        for word in sentence:
            if word in dictionary:
                sentence_id.append(dictionary[word])
            else:
                # sentence_id.append('[UNK]')
                a = np.random.randint(low=0, high=config.vocab_size)
                sentence_id.append(a)
        return sentence_id  # [1,2,3]

    def index_sentence_position_mask(self, original_data):
        '''

        :param original_data: original_data :Dict {head: tail: text: }
        :return: index of sentence,position,and mask
        '''
        sentence = self.tokenizer(original_data['text'])
        assert os.path.exists(config.path_root + config.vocab_path),'you need to build vocab first!'
        dictionary = np.load(config.path_root + config.vocab_path, allow_pickle=True).item()
        sentence_index = self.sentence2index(dictionary, sentence)

        pos1, pos2 = [], []
        entity1 = original_data['head']
        entity2 = original_data['tail']

        ent1pos = original_data['text'].index(entity1)
        ent2pos = original_data['text'].index(entity2)



        for idx, word in enumerate(sentence):
            position1 = self.get_position(idx - ent1pos)
            position2 = self.get_position(idx - ent2pos)

            pos1.append(position1)
            pos2.append(position2)
            # padding
        while len(pos1) < config.max_sentence_length:
            pos1.append(0)
        while len(pos2) < config.max_sentence_length:
            pos2.append(0)
        pos1 = pos1[:config.max_sentence_length]
        pos2 = pos2[:config.max_sentence_length]

        # Mask
        # we need mask to avoid padding part
        '''
        过短的句子可以通过 padding 增加到固定的长度，但是 padding 对
        应的字符只是为了统一长度，并没有实际的价值，因此希望在之后的计算中屏蔽它们，
        这时候就需要 Mask。
        '''
        mask = []
        pos_min = min(ent1pos, ent2pos)
        pos_max = max(ent1pos, ent2pos)
        for i in range(len(sentence_index)):
            if i <= pos_min:
                mask.append(1)
            elif i <= pos_max:
                mask.append(2)
            else:
                mask.append(3)
        # Padding
        while len(mask) < config.max_sentence_length:
            mask.append(0)
        mask = mask[:config.max_sentence_length]

        while len(sentence_index) < config.max_sentence_length:
            sentence_index.append(0)
        sentence_index = sentence_index[:config.max_sentence_length]

        # mask = torch.tensor(mask).long().unsqueeze(0)  # (1, L)
        pos1 = torch.tensor(pos1).long()
        pos2 = torch.tensor(pos2).long()

        sentence_index = torch.tensor(sentence_index).long()
        mask = torch.tensor(mask).long()
        return [sentence_index, pos1, pos2, mask]

    def get_position(self, pos):
        # nn.embedding can be negative number
        '''
        : -limit ~ limit => 0 ~ limit * 2 + 2
        : <-20  => 0
        : -20 => 1
        : 20 => 41
        : >20 => 42
        :param pos:
        :return: positive number
        '''
        if pos < -config.pos_limit:
            return 0
        if - config.pos_limit <= pos <= config.pos_limit:
            return pos + config.pos_limit + 1
        if pos > config.pos_limit:
            return config.pos_limit * 2 + 1

class RelationAndId:
    def __init__(self):
        with open(config.path_root+config.relation_file_path, 'r', encoding='utf-8') as fp:
            json_data = json.load(fp)
        self.json_data = json_data

    def relation2id(self,relation):

        return self.json_data[1][relation]

    def id2relation(self,id):

        return self.json_data[0][id]

if __name__ == '__main__':
    # print("数据预处理开始......")
    # st = SentenceProcess()
    # st.build_vocab(filepath=config.train_data_path,vocabsize=config.vocab_size)
    # print("数据预处理完毕！")

    dictionary = np.load(config.vocab_path, allow_pickle=True).item()
    print(len(dictionary))
