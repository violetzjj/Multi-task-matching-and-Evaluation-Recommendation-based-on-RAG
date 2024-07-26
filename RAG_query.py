from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
from RAG.LLM import OpenAIChat, InternLMChat
from RAG.Embeddings import OpenAIEmbedding
import json
import random
import pandas as pd
import re
from openai import OpenAI
import os
import time

#配置大模型apikey和网址
apikey = os.environ['OPENAI_API_KEY'] = 'your_api key'
baseurl = os.environ['OPENAI_BASE_URL']="openai url"


vector = VectorStore()

vector.load_vector('demo_all1')  #加载本地数据库

embedding = OpenAIEmbedding()


query = input("请输入查询地址：")


#返回向量库中相关的信息
content = vector.query(query,EmbeddingModel = embedding,k=3)
print(f'检索返回内容为：{content}')


def convert_to_decimal(deg, min, sec):

    return float(deg) + float(min) / 60 + float(sec) / 3600

def extract_coordinates(lon_lat):"""从字符串中提取经纬度并转换为十进制"""
    lon, lat = lon_lat
    lon_deg, lon_min, lon_sec = re.findall(r"(\d+)° (\d+)' (\d+\.\d+)""", lon)[0]
    lat_deg, lat_min, lat_sec = re.findall(r"(\d+)° (\d+)' (\d+\.\d+)""", lat)[0]
    lon_decimal = convert_to_decimal(lon_deg, lon_min, lon_sec)
    lat_decimal = convert_to_decimal(lat_deg, lat_min, lat_sec)
    return (lat_decimal, lon_decimal)


def get_matching_addresses(query_address, file_path):
   
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)


    for item in data:
        if query_address in item:
            return item[query_address]

    return None


def get_coordinates(address_list, csv_file_path):

    df = pd.read_csv(csv_file_path)
    coordinates_dict = {}

    for address in address_list:
        matching_row = df[(df['ADDRESS'] == address) | (df['NAME'] == address)]
        if not matching_row.empty:
            latitude = matching_row.iloc[0]['lat']
            longitude = matching_row.iloc[0]['lon']
            coordinates_dict[address] = (latitude,longitude)
        else:
            coordinates_dict[address] = (None, None)

    return coordinates_dict


def convert_dict_to_list(coordinates_dict):
    coordinates_list = []

    for address, (latitude, longitude) in coordinates_dict.items():
        coordinates_list.append([address, latitude, longitude])

    return coordinates_list

def get_coordinates1(address, csv_file_path):

    df = pd.read_csv(csv_file_path)


    matching_row = df[(df['ADDRESS'] == address) | (df['NAME'] == address)]
    if not matching_row.empty:
        latitude = matching_row.iloc[0]['lat']
        longitude = matching_row.iloc[0]['lon']
        return (latitude, longitude)
    else:
        return (None, None)  

#POIsearch
def get_coordinates2(address, csv_file_path):

    df = pd.read_csv(csv_file_path)


    matching_row = df[(df['address'] == address)]
    if not matching_row.empty:
        latitude = matching_row.iloc[0]['lat']
        longitude = matching_row.iloc[0]['lon']
        return (latitude, longitude)
    else:
        return (None, None)


file_path = "GNEMM/Semantic_result.json"    #ca and intersection addresses
file_path1 = "GNEMM/semantic_result2.json"   #poi address
poi_file_path = "GNEMM/weibo_search.csv"
csv_file_path ="GNEMM/Original Shandong_address_data.csv"



query_lat_lon_dec= get_coordinates2(query,poi_file_path)
# query_lat_lon_dec = extract_coordinates(query_lat_lon)   
print(f"查询地址的经纬度信息为{query_lat_lon_dec}")

GNEMM_recommend = get_matching_addresses(query,file_path1)  
GNEMM_lat_lon = get_coordinates(GNEMM_recommend,csv_file_path)  
for address,lat_lon in GNEMM_lat_lon.items():
    GNEMM_lat_lon[address] = extract_coordinates(lat_lon)
print(GNEMM_lat_lon)
GNEMM_lat_lon1 = convert_dict_to_list(GNEMM_lat_lon)
# GNEMM_info = GNEMM_recommend+GNEMM_lat_lon1


prompt = ""
intial_prompt = ""
COT_prompt = ""
zeroshot_prompt =""
self_consistence_prompt =""

#intial prompt
intial_prompt += f"你的任务是根据用户查询地址:{query},综合从Retrieval阶段返回的相关信息和模型生成的top5地址，给出最优的推荐地址.\t"
intial_prompt += f"首先，提取出RAG结果{content}中的地址与{query}相关信息.\t"
intial_prompt += f"我们的模型根据查询地址给出的top5的推荐地址及经纬度信息如下{GNEMM_lat_lon1}\t"
intial_prompt += f"结合匹配结果和提取信息，推荐最接近查询地址的地址"

#prompt1-COT
COT_prompt +=f"任务描述：从给定的候选地址列表中，基于与{query}的相关性，推荐三个最相关的地址。\t"
COT_prompt += f"用户查询的地址为：{query}，查询地址的经纬度为{query_lat_lon_dec}\t"
COT_prompt +=f"我们的模型给出的top5匹配地址及经纬度如下：{GNEMM_lat_lon1}\t"
COT_prompt += f"使用RAG方法从外部地址库中检索得到的相近信息为{content}\t"
COT_prompt +="请你根据下列步骤一步步分析：第一步，提取RAG得到的相近信息中与查询地址有关的地址内容。第二步，分析用户查询，识别查询地址的关键词;第三步，分别从提取出的相近信息和模型的top5地址列表中帅选出包含关键词的地址;第四步，使用地理坐标计算每个地址与查询地址之间的距离第五步，根据距离和关键词匹配度从筛选出的地址进行排序;第六步，选择排名最高的三个地址作为推荐结果;第七步，如果查询地址在列表中，确保它被选为推荐结果之一;第八步，输出推荐的三个地址及其坐标"

#prompt2-Zero-shot-learning
zeroshot_prompt += f"任务描述：从给定的候选地址列表中，基于与{query}的相关性，推荐三个最相关的地址。\t"
zeroshot_prompt +=f"用户查询地址为{query}，查询地址的经纬度为{query_lat_lon_dec}\t"
zeroshot_prompt += f"我们构建的模型的top5的匹配地址及经纬度为{GNEMM_lat_lon1}\t"
zeroshot_prompt +=f"使用RAG方法从外部检索得到的相关信息为{content}"
zeroshot_prompt += f"解释任务：用户正在寻找与查询地址相关的地址。识别关键信息：识别用户查询地址中包含的关键地址元素，关键词是推荐地址的相关性标准。筛选地址：从模型候选地址和外部检索的相关信息中筛选出包含关键地址元素的地址，因为它们与用户查询直接相关。地理邻近性：考虑每个筛选出的地址与查询地址的地理距离，优先选择距离近的地址。推荐排序：根据地址与查询地址的地理邻近性和关键词匹配度进行排序，如果查询地址在列表中，确保它被选为推荐结果之一。输出推荐：选择排序最高的三个地址作为推荐结果。"


self_consistence_prompt += f"任务描述：从给定的候选地址列表中，基于与{query}的相关性，推荐三个最相关的地址。\t"
self_consistence_prompt += f"用户查询地址为{query}，查询地址的经纬度为{query_lat_lon_dec}\t"
self_consistence_prompt += f"我们构建的模型的top5的匹配地址及经纬度为{GNEMM_lat_lon1}\t"
self_consistence_prompt += f"使用RAG方法从外部检索得到的相关信息为{content}"
self_consistence_prompt += "1.提取RAG返回的检索信息中与查询地址相关的信息，作为候选地址列表A。2.模型的top5地址列表同样为可以作为候选地址列表B。3.对于每个候选地址，请生成多个推理路径，解释为什么这个地址是合适的推荐。4.请记住，一个地址可能有多个有效的推理路径，每条路径都可能导致不同的推荐。5.在生成推理路径后，请为每个地址提供一个最终答案（即推荐或不推荐）。6.从每个地址生成的所有推理路径中，选择出现次数最多的答案作为最一致的推荐。7.对于每个地址，执行以下步骤：a. 生成多个推理路径，每个路径都基于地址的名称和坐标。b. 对于每个推理路径，评估它与用户查询的一致性。c. 记录每个地址作为推荐答案的一致性得分。最终，为每个地址计算一致性得分，并选择得分最高的三个地址作为推荐。"
# print(prompt)


#Generation
# Initialize OpenAI client
client = OpenAI(api_key=apikey, base_url=baseurl)



#generate results
start_time = time.time()
response = client.chat.completions.create(model = "gpt-3.5-turbo", messages=[{"role": "system", "content": COT_prompt}], temperature=0.3)
end_time = time.time()
consume_time = end_time-start_time

print(f"本次生成耗时{consume_time}")
print(f"模型推荐的最终地址为{response.choices[0].message.content}")