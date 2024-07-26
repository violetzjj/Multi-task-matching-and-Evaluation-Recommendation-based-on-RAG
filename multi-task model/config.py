import os
import torch

data_dir = os.getcwd() + '/data/clue/'
train_dir = data_dir + 'train.npz'
dev_dir = data_dir + 'dev.npz'
test_dir = data_dir + 'test.npz'
files = ['train', 'dev', 'test']
bert_model = 'pretrained_bert_models/bert-base-chinese/'
roberta_model = 'pretrained_bert_models/chinese_roberta_wwm_large_ext/'
macbert_model = 'pretrained_bert_models/chinese-electra-base/'
model_dir = os.getcwd() + '/experiments/clue/'
log_dir = model_dir + 'train.log'
case_dir = os.getcwd() + '/case/bad_case.txt'

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 64
epoch_num = 30
min_epoch_num = 5
patience = 0.0002
patience_num = 10

# lstm_embedding_size = 1024

gpu = 0

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")

labels = ['1', '2', '3']

label2id = {
    '1': 0,
    "2": 1,
    "3": 2,
}

id2label = {_id: _label for _label, _id in list(label2id.items())}
