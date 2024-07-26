from data_loader import MultiTaskDataset
from torch.utils.data import DataLoader
from model import BertMultiTask
import config
from train import predict_vector
from scipy import spatial
import pickle
import heapq
import sys
import codecs

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
def custom_distance(address1, address2):
    word_test = []
    address1_list = list(address1)
    address1_list_len = len(address1_list)
    address2_list = list(address2)
    address2_list_len = len(address2_list)
    if address1_list_len <100:
        address1_list = address1_list + ['M'] * (100-address1_list_len)
    if address2_list_len <100:
        address2_list = address2_list + ['M'] * (100-address2_list_len)
    word_test.append(address1_list)
    word_test.append(address2_list)
    label_test = []
    label1_list = ['1'] * address1_list_len
    label2_list = ['1'] * address2_list_len
    label_test.append(label1_list)
    label_test.append(label2_list)
    label_match_test = [1, 1]
    label_score_test = ['5.0', '5.0']
    test_dataset = MultiTaskDataset(word_test, label_test, label_match_test, label_score_test, config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    model = BertMultiTask.from_pretrained(config.model_dir)
    model.to(config.device)
    output_score = predict_vector(test_loader, model, mode='test')
    cos_sim = 1 - spatial.distance.cosine(output_score[0][0], output_score[0][1])
    return cos_sim

def compute_address_vector(data):
    word_test = []
    label_test = []
    label_match_test = []
    label_score_test = []
    for address in data:
        address_list = list(address)
        address_list_len = len(address_list)
        if address_list_len < 100:
            address_list = address_list + ['M'] * (100 - address_list_len)
        word_test.append(address_list)
        label_list = ['1'] * address_list_len
        label_test.append(label_list)
        label_match_test.append(1)
        label_score_test.append('5.0')
    test_dataset = MultiTaskDataset(word_test, label_test, label_match_test, label_score_test, config)
    print("--------Dataset Build!--------")
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    print("--------Get Dataloader!--------")
    model = BertMultiTask.from_pretrained(config.model_dir)
    model.to(config.device)
    print("--------Start Computing!--------")
    output_vectors = predict_vector(test_loader, model, mode='test')
    return output_vectors

def flat_list(lst):
    res = []
    for sub_lst in lst:
        for item in sub_lst:
            res.append(item)
    return res


# test_address1 = "香港路与合肥路交汇处向南300米路北正西方向10米"
# test_address2 = "竹亭饭店"
# test_vector = custom_distance(test_address1, test_address2)
# print(test_vector)

data_fenge_list = []
data_total_list = []
address_set_total = set()
count = 0

with open('shandongaddress_dataset.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    data_block = []
    index = 0
    for line in lines:
        line = line.replace('\n', '')
        if line in address_set_total:
            continue
        count += 1
        address_set_total.add(line)
        data_block.append(line)
        if count == 300000:
            count = 0
            with open('shandongaddress/data_' + str(index) + '.pkl', 'wb') as f:
                pickle.dump(data_block, f)
            index += 1
            data_block = []

if count != 0:
    with open('shandongaddress/data_' + str(index) + '.pkl', 'wb') as f:
        pickle.dump(data_block, f)
        index += 1
        
for i in range(index):
    with open('shandongaddress/data_' + str(i) + '.pkl', 'rb') as f:
        temp = pickle.load(f)
        data_fenge_list.append(temp)
        data_total_list.extend(temp)

for i in range(len(data_fenge_list)):
    print('block ' + str(i+1))
    vectors = compute_address_vector(data_fenge_list[i])
    with open('shandongaddress/vectors_' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(vectors, f)

vectors_total_list = []
for i in range(len(data_fenge_list)):
    with open('shandongaddress/vectors_' + str(i) + '.pkl', 'rb') as f:
        vectors_total_list.extend(pickle.load(f))


vectors_loaded_flat = flat_list(vectors_total_list)

similarity_scores = []

query = "蓝天国贸中心603"
query_vector = compute_address_vector([query])[0]
for i in range(len(vectors_loaded_flat)):
    cos_sim = 1 - spatial.distance.cosine(vectors_loaded_flat[i], query_vector)
    similarity_scores.append(cos_sim)

n = 10
largest_n_with_index = heapq.nlargest(n, enumerate(similarity_scores), key=lambda x: x[1])

for item in largest_n_with_index:
    index = item[0]
    score = item[1]
    print(data_total_list[index], score)


