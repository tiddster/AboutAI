from py2neo import *

math_graph = Graph('http://localhost:7474', auth=('neo4j', '123456'))

integration = Node("主概念", name="积分")
integrable = Node("属性", name="可积分")
differentialEquations = Node("其他主概念", name="微分方程")

math_graph.create(Relationship(integration, "强相关于",differentialEquations))
math_graph.create(Relationship(integration, "属性", integrable))

basicIntegration = ["定积分","不定积分"]
special = ["反常积分","有理函数的积分"]
formulaOfBI = ["牛顿-莱布尼茨公式"]
method = ["换元积分法", "分部积分法"]
multipleIntegration = ["二重积分", "三重积分"]
curveIntegration = ["曲线积分","曲面积分"]
formulaOfIntegration = [["格林公式"], ["高斯公式", "斯托克斯公式"]]

for i in range(len(special)):
    A = Node("概念", name=special[i])
    B = Node("概念", name=basicIntegration[i])
    math_graph.create(Relationship(B, "包含", A))
    math_graph.create(Relationship(integration,"包含",B))

for s in formulaOfBI:
    matcher = NodeMatcher(math_graph)
    node = matcher.match("概念",name="定积分").first()
    nodeF = Node("公式", name=s)
    math_graph.create(Relationship(node, "常用公式", nodeF))


for i in range(len(multipleIntegration)):
    matcher = NodeMatcher(math_graph)
    node = matcher.match("概念",name="定积分").first()
    A = Node("概念", name=multipleIntegration[i], chapter=10)
    B = Node("概念", name=curveIntegration[i], chapter=11)
    C:Node
    D = Node("基本方法", name=method[i])
    for j in range(len(formulaOfIntegration[i])):
        C = Node("公式", name=formulaOfIntegration[i][j], chapter=11)
        math_graph.create(Relationship(C, "公式用于", B))
    math_graph.create(Relationship(A, "拓展出", B))
    math_graph.create(Relationship(integration, "方法有", D))
    math_graph.create(Relationship(node, "包含", A))


# 查找标签 全部
print(f"所有概念：{math_graph.nodes.match('概念').all()}")

# 查找标签 第一个
print(f"所有方法：{math_graph.nodes.match('方法有').first()}")

# 根据属性查找
print(f"所有11章内容: {math_graph.nodes.match().where(chapter=11).all()}")

# 输出匹配的个数
print(f"概念的个数: {math_graph.nodes.match('概念').count()}")

print(f"关于积分的方法关系: {math_graph.relationships.match([integration], '方法有').all()}")
