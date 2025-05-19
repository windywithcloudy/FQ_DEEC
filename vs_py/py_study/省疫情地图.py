import json
from pyecharts.charts import Map
from pyecharts.options import *

f = open("E:\BaiduNetdiskDownload\资料\资料\可视化案例数据\地图数据\\疫情.txt",'r',encoding="UTF-8")
map_data = json.loads(f.read())
f.close()

province_data = map_data['areaTree'][0]['children'][3]['children']
data_list=[]
for city in province_data:
    city_name = city['name']
    city_confirm_num = city['total']['confirm']
    city_name +='市'
    data_list.append((city_name,city_confirm_num))
data_list.append(("济源市",5))
map = Map()
map.add("河南省各市确诊人数",data_list,'河南')
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
map.render("河南省疫情地图.html")