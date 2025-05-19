import json
import pyecharts
from pyecharts.options import TitleOpts,LegendOpts,ToolboxOpts,VisualMapOpts,LabelOpts
f = open("E:\BaiduNetdiskDownload\资料\资料\可视化案例数据\折线图数据\\美国.txt",'r',encoding='UTF-8')
j = open("E:\BaiduNetdiskDownload\资料\资料\可视化案例数据\折线图数据\\日本.txt",'r',encoding='UTF-8')
i = open("E:\BaiduNetdiskDownload\资料\资料\可视化案例数据\折线图数据\\印度.txt",'r',encoding='UTF-8')
us_data = f.read()
jp_data = j.read()
in_data = i.read()
us_data = us_data.replace('jsonp_1629344292311_69436(',' ')
us_data = us_data[:-2]
jp_data = jp_data.replace('jsonp_1629350871167_29498(',' ')
jp_data = jp_data[:-2]
in_data = in_data.replace('jsonp_1629350745930_63180(',' ')
in_data = in_data[:-2]

us_ready_data = json.loads(us_data)
jp_ready_data = json.loads(jp_data)
in_ready_data = json.loads(in_data)
us_trend = us_ready_data['data'][0]['trend']
us_x_dates = us_trend["updateDate"][:314]
us_y_data = us_trend['list'][0]["data"][:314]
jp_trend = jp_ready_data['data'][0]['trend']
jp_x_dates = jp_trend["updateDate"][:314]
jp_y_data = jp_trend['list'][0]["data"][:314]
in_trend = in_ready_data['data'][0]['trend']
in_x_dates = in_trend["updateDate"][:314]
in_y_data = in_trend['list'][0]["data"][:314]
line = pyecharts.charts.Line()
line.add_xaxis(us_x_dates)
line.add_yaxis("美国确诊人数",us_y_data,label_opts=LabelOpts(is_show = False))
line.add_yaxis("日本确诊人数",jp_y_data,label_opts=LabelOpts(is_show = False))
line.add_yaxis("印度确诊人数",in_y_data,label_opts=LabelOpts(is_show = False))
line.set_global_opts(                                                          #配置全局属性
    title_opts = TitleOpts(title="疫情数据展示",pos_left="center",pos_bottom="1%"),  #配置标题
    legend_opts=LegendOpts(is_show=True),                                       #配置图例
    toolbox_opts=ToolboxOpts(is_show=True),                                     #配置工具箱
    #visualmap_opts=VisualMapOpts(is_show=True)                                  #配置视觉图
)

line.render()

f.close()
j.close()
i.close()