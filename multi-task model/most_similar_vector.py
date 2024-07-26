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
    with open('vector_dataset.pkl', 'wb') as f:
      pickle.dump(test_dataset, f)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    print("--------Get Dataloader!--------")
    with open('vector_loader.pkl', 'wb') as f:
      pickle.dump(test_loader, f)
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

data = []

with open('shandongaddress_dataset.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '')
        data.append(line)

with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)

with open('data.pkl', 'rb') as f:
    data_loaded = pickle.load(f)

vectors = compute_address_vector(data_loaded)

with open('vectors.pkl', 'wb') as f:
    pickle.dump(vectors, f)

with open('vectors.pkl', 'rb') as f:
    vectors_loaded = pickle.load(f)


vectors_loaded_flat = flat_list(vectors_loaded)

similarity_scores = []

query = "北大街53号华夏传媒大厦2F北区3F"
query_vector = compute_address_vector([query])[0]
for i in range(len(vectors_loaded_flat)):
    cos_sim = 1 - spatial.distance.cosine(vectors_loaded_flat[i], query_vector)
    similarity_scores.append(cos_sim)

n = 5
largest_n_with_index = heapq.nlargest(n, enumerate(similarity_scores), key=lambda x: x[1])

for item in largest_n_with_index:
    index = item[0]
    score = item[1]
    print(data_loaded[index], score)


