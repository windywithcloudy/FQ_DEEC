from pyecharts.charts import Bar,Timeline
from pyecharts.options import LabelOpts
from pyecharts.globals import ThemeType

bar1 = Bar()
bar1.add_xaxis(["china",'America','japan'])
bar1.add_yaxis("GDP",[30,20,10],label_opts=LabelOpts(position="right"))
#反转x，y轴
bar1.reversal_axis()

bar2 = Bar()
bar2.add_xaxis(["china",'America','japan'])
bar2.add_yaxis("GDP",[50,30,15],label_opts=LabelOpts(position="right"))
#反转x，y轴
bar2.reversal_axis()

bar3 = Bar()
bar3.add_xaxis(["china",'America','japan'])
bar3.add_yaxis("GDP",[70,40,20],label_opts=LabelOpts(position="right"))
#反转x，y轴
bar3.reversal_axis()

timeline = Timeline({"theme":ThemeType.LIGHT})
timeline.add(bar1,'piont1')
timeline.add(bar2,'piont2')
timeline.add(bar3,'piont3')

timeline.add_schema(
    play_interval=500,
    is_timeline_show=True,
    is_auto_play=True,
    is_loop_play=True
)

timeline.render("基础动态柱状图.html")