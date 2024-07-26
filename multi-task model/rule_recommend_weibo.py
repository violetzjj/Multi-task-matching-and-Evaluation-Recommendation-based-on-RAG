from data_loader import MultiTaskDataset
from torch.utils.data import DataLoader
from model import BertMultiTask
import config
from train import predict_vector
from scipy import spatial
import pickle
import heapq
import time
from datetime import timedelta
import sys
import codecs
import json
import re
import pandas as pd
from math import sin, asin, cos, radians, fabs, sqrt, pow

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

start_time = time.time()

EARTH_RADIUS = 6371  # The average radius of the earth, 6371km


def hav(theta):
    s = sin(theta / 2)
    return s * s


def takeThird(elem):
    return elem[2]


def get_distance_hav(lat0, lng0, lat1, lng1):
    "Use the Haversine formula to calculate the distance between two points on the sphere."
    # Longitude and latitude converted to radians
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))

    return distance


addressDict = {}


def dms_to_decimal(dms):
    degree = float(dms.split('°')[0])
    minute = float(dms.split('°')[1].split('\'')[0])
    second = float(dms.split('°')[1].split('\'')[1].split('""')[0])
    return degree + minute / 60 + second / 3600


df = pd.read_csv('shandongaddress.csv', encoding='utf-8')
for index, row in df.iterrows():
    if isinstance(row['ADDRESS'], str):
        address = row['ADDRESS']
    else:
        address = row['NAME']
    address = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', address)
    lon = row['lon']
    lat = row['lat']
    if not isinstance(lon, str):
        continue
    if not isinstance(lat, str):
        continue
    lon_num = dms_to_decimal(lon)
    lat_num = dms_to_decimal(lat)
    addressDict[address] = {}
    addressDict[address]['lon'] = lon_num
    addressDict[address]['lat'] = lat_num

address_media_dict = {}
with open('jinan_address_clean_remove_duplicate.csv', mode='r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '')
        address = line.split(',')[0]
        lon = float(line.split(',')[1])
        lat = float(line.split(',')[2])
        address_media_dict[address] = {}
        address_media_dict[address]['lon'] = lon
        address_media_dict[address]['lat'] = lat


def custom_distance(address1, address2):
    word_test = []
    address1_list = list(address1)
    address1_list_len = len(address1_list)
    address2_list = list(address2)
    address2_list_len = len(address2_list)
    if address1_list_len < 100:
        address1_list = address1_list + ['M'] * (100 - address1_list_len)
    if address2_list_len < 100:
        address2_list = address2_list + ['M'] * (100 - address2_list_len)
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


data_fenge_list = []
data_total_list = []
count = 0

index = 5
for i in range(index):
    with open('shandongaddress/data_' + str(i) + '.pkl', 'rb') as f:
        temp = pickle.load(f)
        data_fenge_list.append(temp)
        data_total_list.extend(temp)

keys = []
keys_predicted = []
with open("Top5Predict_weibo_rule.json", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '')
        data = json.loads(line)
        key = list(data.keys())[0]
        keys_predicted.append(key)

with open("shandongaddressTop5Label_weibo.json", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '')
        data = json.loads(line)
        key = list(data.keys())[0]
        if key in keys_predicted:
            continue
        keys.append(key)

for query in keys:
    query_vector = compute_address_vector([query])[0]
    similarity_scores = []
    for i in range(len(data_fenge_list)):
        vectors_total_list = []
        with open('shandongaddress/vectors_' + str(i) + '.pkl', 'rb') as f:
            vectors_total_list.extend(pickle.load(f))
            vectors_loaded_flat = flat_list(vectors_total_list)
            for j in range(len(vectors_loaded_flat)):
                cos_sim = 1 - spatial.distance.cosine(vectors_loaded_flat[j], query_vector)
                similarity_scores.append(cos_sim)

    n = len(similarity_scores)
    largest_n_with_index = heapq.nlargest(n, enumerate(similarity_scores), key=lambda x: x[1])

    query_lon = address_media_dict[query]['lon']
    query_lat = address_media_dict[query]['lat']

    result_list = []
    count = 0
    for item in largest_n_with_index:
        idx = item[0]
        score = item[1]
        if data_total_list[idx] == query:
            continue
        try:
        	temp_lon = addressDict[data_total_list[idx]]['lon']
        	temp_lat = addressDict[data_total_list[idx]]['lat']
        except:
          continue
        if get_distance_hav(query_lat, query_lon, temp_lat, temp_lon) > 3:
            continue
        count += 1
        result_list.append((data_total_list[idx], score))
        if count == 5:
            break
    result_dict = {query: result_list}
    with open("Top5Predict_weibo_rule.json", "a", encoding='utf-8') as outfile:
        json.dump(result_dict, outfile, ensure_ascii=False)
        outfile.write('\n')

end_time = time.time()
elapsed_time = end_time - start_time
print("程序运行时间：", timedelta(seconds=elapsed_time))
