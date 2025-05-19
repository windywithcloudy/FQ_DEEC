import pyecharts
from pyecharts.options import TitleOpts,LegendOpts,ToolboxOpts,VisualMapOpts
line = pyecharts.charts.Line()
line.add_xaxis(["中国","美国","英国"])
line.add_yaxis("GDP",[30,20,10])
line.set_global_opts(                                                          #配置全局属性
    title_opts = TitleOpts(title="GDP展示",pos_left="center",pos_bottom="1%"),  #配置标题
    legend_opts=LegendOpts(is_show=True),                                       #配置图例
    toolbox_opts=ToolboxOpts(is_show=True),                                     #配置工具箱
    visualmap_opts=VisualMapOpts(is_show=True)                                  #配置视觉图
)
line.render()