import json
from pyecharts.charts import Map
from pyecharts.options import *

f = open("E:\BaiduNetdiskDownload\资料\资料\可视化案例数据\地图数据\\疫情.txt",'r',encoding="UTF-8")
map_data = json.loads(f.read())
f.close()

special_provinces = {"北京": "市", "天津": "市", "上海": "市", "重庆": "市", "香港": "特别行政区",
"澳门": "特别行政区", "新疆": "维吾尔自治区", "西藏": "自治区", "内蒙古": "自治区",
"广西": "壮族自治区", "宁夏": "回族自治区"}

province_data = map_data['areaTree'][0]['children']
data_list =[ ]
for province in province_data:
    province_name = province['name']
    province_confirm_data = province['total']['confirm']   
    # 判断是否为直辖市或特别行政区
    if province_name in special_provinces:
        province_name += special_provinces[province_name]
    else:
        province_name += "省"
    data_list.append((province_name,province_confirm_data))


map = Map()
map.add("各省份确诊人数",data_list,'china')
map.set_global_opts(
    title_opts=TitleOpts(title="全国疫情地图"),
    visualmap_opts=VisualMapOpts(
        is_show=True,
        is_piecewise=True,
        pieces=[
            {"min":1,"max":9,"label":"1-9人","color":"#CCFFFF"},
            {"min":10,"max":99,"label":"10-99人","color":"#ddFFFF"},
            {"min":100,"max":499,"label":"100-499人","color":"#ddeeaF"},
            {"min":500,"max":999,"label":"500-999人","color":"#ae3fca"},
            {"min":1000,"max":4999,"label":"1000-4999人","color":"#125acf"},
            {"min":5000,"max":9999,"label":"5000-9999人","color":"#abcabc"},
            {"min":10000,"label":"10000+人","color":"#cfcfcf"},
        ]
    )
)

map.render("全国疫情地图.html")