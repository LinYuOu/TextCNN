import torch
from torch.utils.data import Dataset
from Exp3_DataProc import *
import torch.nn as nn
from Exp3_Config import Training_Config
# 训练集和验证集
config = Training_Config()
class TextDataSet(Dataset):
    def __init__(self, filepath):
        lines = open(filepath, 'r', encoding='utf-8').readlines()
        self.original_data = []
        for line in lines:
            tmp = {}
            line = line.split('\t')
            tmp['head'] = line[0]
            tmp['tail'] = line[1]

            tmp['relation'] = line[2]
            tmp['text'] = line[3][:-1]
            self.original_data.append(tmp)

    def __getitem__(self, index):
        '''

        :self.original_data: Dict {head: tail: text: }
        :return: index of sentence position and mask
        '''
        senProcess = SentenceProcess()
        # sentence_index, pos1, pos2,mask = senProcess.index_sentence_position_mask(self.original_data)
        seq = senProcess.index_sentence_position_mask(self.original_data[index])
        relation_id = RelationAndId()
        label = relation_id.relation2id(self.original_data[index]['relation'])
        return seq, label

    def __len__(self):
        return len(self.original_data)


# 测试集是没有标签的，因此函数会略有不同
class TestDataSet(Dataset):
    def __init__(self, filepath):
        lines = open(filepath, 'r', encoding='utf-8').readlines()
        self.original_data = []
        for line in lines:
            tmp = {}
            line = line.split('\t')
            tmp['head'] = line[0]
            tmp['tail'] = line[1]
            tmp['text'] = line[2][:-1]
            self.original_data.append(tmp)

    def __getitem__(self, index):
        '''

                :self.original_data: Dict {head: tail: text: }
                :return: index of sentence position and mask
                '''
        senProcess = SentenceProcess()
        # sentence_index, pos1, pos2,mask = senProcess.index_sentence_position_mask(self.original_data)
        seq = senProcess.index_sentence_position_mask(self.original_data[index])
        self.len = len(seq[0])
        return seq

    def __len__(self):
        return len(self.original_data)


if __name__ == "__main__":
    trainset = TextDataSet(filepath="./data/data_train_new.txt")
    testset = TestDataSet(filepath="./data/test_exp3.txt")
    # print("训练集长度为：", len(trainset))
    # print("测试集长度为：", len(testset))

