from pyecharts.charts import Bar
from pyecharts.options import LabelOpts
bar = Bar()

bar.add_xaxis(["china",'America','japan'])
bar.add_yaxis("GDP",[30,20,10],label_opts=LabelOpts(position="right"))
#反转x，y轴
bar.reversal_axis()
bar.render("柱状图试学.html")
