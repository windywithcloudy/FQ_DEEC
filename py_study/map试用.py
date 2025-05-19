from pyecharts.charts import Map
from pyecharts.options import VisualMapOpts
map = Map()

data = [                        #要打全省与市名
    ("北京市",99),
    ("上海市",199),
    ("湖南省",299),
    ("台湾省",199),
    ("广东省",99)
]

map.add("测试地图",data,"china")
map.set_global_opts(
    visualmap_opts=VisualMapOpts(
        is_show=True,
        is_piecewise=True,                                      #开启手动配置范围
        pieces=[
            {"min":1,"max":9,"label":"1-9人","color":"#CCFFFF"},
            {"min":10,"max":99,"label":"10-99人","color":"#ddFFFF"},
            {"min":100,"max":199,"label":"100-199人","color":"#ddeeaF"},
            {"min":200,"max":499,"label":"200-499人","color":"#ae3fca"}
        ]
    )
)
map.render()