from pyecharts import options as opts
from pyecharts.charts import Graph

import pyecharts
print(pyecharts.__version__)

#创建点
nodes = [
    {"name": "1", "symbolValue": "10"},
    {"name": "2", "symbolValue": "20"},
    {"name": "3", "symbolValue": "30"},
    {"name": "4", "symbolValue": "40"},
    {"name": "5", "symbolValue": "50"},
    {"name": "6", "symbolValue": "60"},
    {"name": "7", "symbolValue": "70"},
    {"name": "8", "symbolValue": "80"},
    {"name": "9", "symbolValue": "90"}
]

#创建边
links = []
for i in nodes:
    for j in nodes:
        links.append({"source":i.get("name"), "target":j.get("name")})

#创建图
demo1 = (
    Graph()
    .add("",nodes, links, repulsion=8000)
    .set_global_opts(title_opts=opts.TitleOpts(title="demo1"))
    .render("demo1.html")
)