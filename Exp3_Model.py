import torch
import torch.nn as nn
import torch.nn.functional as functions
import torch.nn.functional as F
from Exp3_Config import Training_Config
config = Training_Config()

class TextCNN_Model(nn.Module):

    def __init__(self):
        super(TextCNN_Model, self).__init__()
        # hyper_parameters
        self._minus = -100
        self.act = F.relu
        self.hidden_size = config.filter_num * 3
        self.input_size = config.word_dim + config.pos_dim * 2
        self.kernel_size = 3
        self.padding_size = 1
        self.label_num = config.label_num

        # 词嵌入
        self.word_embedding = nn.Embedding(config.vocab_size, config.word_dim)
        self.pos1_embedding = nn.Embedding(config.pos_limit * 2 + 2 + 1, config.pos_dim)
        self.pos2_embedding = nn.Embedding(config.pos_limit * 2 + 2 + 1, config.pos_dim)

        # encoding sentence level feature via cnn
        self.conv = nn.Conv1d(self.input_size, self.hidden_size, self.kernel_size, padding=self.padding_size)
        # n_channels=1, out_channels=filters_num

        self.dropout = nn.Dropout(config.dropout)
        self.pool = nn.MaxPool1d(config.max_sentence_length)

        self.out_linear = nn.Linear(3 * self.hidden_size,self.label_num)


        # we need mask to avoid padding part
        '''
        过短的句子可以通过 padding 增加到固定的长度，但是 padding 对
        应的字符只是为了统一长度，并没有实际的价值，因此希望在之后的计算中屏蔽它们，
        这时候就需要 Mask。
        '''
        self.mask_embedding = nn.Embedding(4, 3)
        self.mask_embedding.weight.data.copy_(torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.mask_embedding.weight.requires_grad = False



    def forward(self, data):
        '''

        :param token: (BatchSize, Length), index of tokens
        :param pos1: (BatchSize, Length), relative position to head entity
        :param pos2: (BatchSize, Length), relative position to tail entity
        :return:
                (B, EMBED), representations for sentences
        '''
        token, pos1, pos2, mask = data


        # print("token{}\npos1{}\npos2{}\nmask{}\n".format(token.shape, pos1.shape, pos2.shape, mask.shape))
        # Check size of tensors
        if len(token.size()) != 2 or token.size() != pos1.size() or token.size() != pos2.size():
            #  保证token 和pos1,pos2都是 (BatchSize, Length)
            raise Exception("Size of token, pos1 ans pos2 should be (B, L)")
        x = torch.cat([self.word_embedding(token),
                       self.pos1_embedding(pos1),
                       self.pos2_embedding(pos2)], 2)  # (B, L, EMBED)
        x = x.transpose(1, 2)  # (B, EMBED, L)
        # print(x.shape)
        x = self.conv(x)  # (B, H, L)

        mask = 1 - self.mask_embedding(mask).transpose(1, 2)  # (B, L) -> (B, L, 3) -> (B, 3, L)
        # print('mask',mask.shape)
        # print('x',x.shape)
        pool1 = self.pool(self.act(x + self._minus * mask[:, 0:1, :]))  # (B, H, 1)
        pool2 = self.pool(self.act(x + self._minus * mask[:, 1:2, :]))
        pool3 = self.pool(self.act(x + self._minus * mask[:, 2:3, :]))
        x = torch.cat([pool1, pool2, pool3], 1)  # (B, 3H, 1)
        x = x.squeeze(2)  # (B, 3H)
        x = self.dropout(x)

        x = self.out_linear(x)  # (B, label_num)

        return x

