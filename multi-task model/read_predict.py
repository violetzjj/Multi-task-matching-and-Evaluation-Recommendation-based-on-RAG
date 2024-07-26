import json

# 假设您的JSON数据存储在名为data.json的文件中
with open('shandongaddressTop5Label.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 初始化一个空列表来存储查询地址
query_addresses = []

# 遍历JSON数据中的所有项
for address, predictions in data.items():
    # 将查询地址添加到列表中
    query_addresses.append(address)

# 打印查询地址
for address in query_addresses:
    print(address)


# 将查询地址写入到一个文本文件中
with open('query_addresses.txt', 'w', encoding='utf-8') as f:
    for address in query_addresses:
        f.write(address + '\n')