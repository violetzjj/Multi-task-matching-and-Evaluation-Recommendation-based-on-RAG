import re
from math import sin, asin, cos, radians, fabs, sqrt, pow

EARTH_RADIUS = 6371  # The average radius of the earth, 6371km


def hav(theta):
    s = sin(theta / 2)
    return s * s


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


def convert_to_decimal(deg, min, sec):
    """将度、分、秒转换为十进制格式"""
    return float(deg) + float(min) / 60 + float(sec) / 3600

def extract_lat(lat):
    """从字符串中提取经纬度并转换为十进制"""
    lat = lat
    lat_deg, lat_min, lat_sec = re.findall(r"(\d+)° (\d+)' (\d+\.\d+)""", lat)[0]
    lat_decimal = convert_to_decimal(lat_deg, lat_min, lat_sec)
    return lat_decimal

def extract_lon(lon):
    """从字符串中提取经纬度并转换为十进制"""
    lon = lon
    lon_deg, lon_min, lon_sec = re.findall(r"(\d+)° (\d+)' (\d+\.\d+)""", lon)[0]
    lon_decimal = convert_to_decimal(lon_deg, lon_min, lon_sec)
    return lon_decimal



lon1 = input("输入地址1的经度")
print(lon1)
lat1 = input("输入地址1的纬度")
print(lat1)

lon2 = input("输入地址1的经度")
print(lon2)
lat2 = input("输入地址1的纬度")
print(lat2)



distance1_2 = get_distance_hav(lat1,lon1,lat2,lon2)
print(f"两个地址之间的半正矢距离为：{distance1_2}")