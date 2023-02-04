"""
该文件旨在配置训练过程中的各种参数
请按照自己的需求进行添加或者删除相应属性
"""


class Training_Config(object):
    def __init__(self,
                 word_dim=200,
                 vocab_size=3000,
                 training_epoch=200,
                 num_val=2,
                 max_sentence_length=80,
                 cuda=False,
                 label_num=44,
                 learning_rate=0.01,
                 batch_size=162,
                 dropout=0.5,
                 pos_limit=80,
                 pos_dim=2,
                 filter_num=230,
                 path_root='',
                 vocab_path = 'data/vocab.npy',
                 relation_file_path = './data/rel2id.json',
                 train_data_path = 'data/data_train.txt',
                 val_data_path = 'data/data_val.txt',
                 test_data_path = 'data/test_exp3.txt',
                 test_result_path = 'data/exp3_predict_labels.txt'):
        self.word_dim = word_dim  # 词向量的维度
        self.vocab_size = vocab_size  # 词汇表大小
        self.epoch = training_epoch  # 训练轮数
        self.num_val = num_val  # 经过几轮才开始验证
        self.max_sentence_length = max_sentence_length  # 句子最大长度
        self.label_num = label_num  # 分类标签个数
        self.lr = learning_rate  # 学习率
        self.batch_size = batch_size  # 批大小
        self.cuda = cuda  # 是否用CUDA
        self.dropout = dropout  # dropout概率
        self.pos_limit = max_sentence_length  # position range (-limit,limit)
        self.pos_dim = pos_dim  # position 的维度
        self.filter_num = filter_num # 卷积核的个数，230
        self.path_root = path_root # 根目录，本地为‘’
        self.vocab_path = vocab_path
        self.relation_file_path = relation_file_path
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.test_result_path = test_result_path


