from pyecharts.charts import Bar,Timeline
from pyecharts.options import LabelOpts
from pyecharts.globals import ThemeType

f = open("E:\BaiduNetdiskDownload\资料\资料\可视化案例数据\动态柱状图数据\\1960-2019全球GDP数据.csv",'r',encoding='GB2312')
GDP_data = f.readlines()
f.close()
GDP_data.pop(0)

data_dict = {}
for line in GDP_data:
    year = int(line.split(',')[0])
    country = line.split(',')[1]
    gdp = float(line.split(',')[2])
    try:
        data_dict[year].append((country,gdp))
    except KeyError:
        data_dict[year] = []
        data_dict[year].append((country,gdp))

time_line = Timeline({"theme" : ThemeType.LIGHT})
sorted_year = sorted(data_dict.keys())
for year in sorted_year:
    data_dict[year].sort(key = lambda element:element[1],reverse=True)
    year_data = data_dict[year][0:8]
    x_data = []
    y_data = []
    for contry_gdp in year_data:
        x_data.append(contry_gdp[0])
        y_data.append(contry_gdp[1]/100000000)
    
    bar = Bar()
    x_data.reverse()
    y_data.reverse()
    bar.add_xaxis(x_data)
    bar.add_yaxis("GDP(亿)",y_data,label_opts=LabelOpts(position="right"))
    bar.reversal_axis()

    time_line.add(bar,str(year))

time_line.add_schema(
    play_interval=1000,
    is_timeline_show=False,
    is_auto_play=True,
    is_loop_play=True
)

time_line.render("全球GDP变化.html")