from number import Number
from graphviz import Graph
from pathlib import Path

pwd = Path.cwd()
dir = Path.joinpath(pwd, 'graph', 'graph_o2.gv')


class CustomGraph:
    def create(root):
        nodes, edges = set(), set()

        def build(root):
            nodes.add(root)
            for node in root._prev:
                nodes.add(node)
                edges.add((node, root))
                build(node)

        build(root)
        return nodes, edges

    def draw(root):
        nodes, edges = CustomGraph.create(root)

        graph = Graph(format='jpeg', graph_attr={'rankdir': 'LR'})

        for node in nodes:
            graph.node(name=str(id(node)), label="{%s | data: %.4f | grad: %.4f}" % (
                node.label, node.data, node.grad), shape='record')

        for head, tail in edges:
            graph.edge(str(id(head)), str(id(tail)))

        graph.render(dir)


# input x1, x2
x1 = Number(data=2.0, label='x1')
x2 = Number(data=0.0, label='x2')
# weights w1, w2
w1 = Number(data=-3.0, label='w1')
w2 = Number(data=1.0, label='w2')
# bias of the neutron
b = Number(data=6.8813735870195432, label='b')

x1w1 = x1 * w1
x1w1.label = 'x1*w1'
x2w2 = x2 * w2
x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = 'x1*w1+x2*w2'

n = x1w1x2w2 + b
n.label = 'n'

o = n.tanh()
o.label = 'o'

o.backward()

CustomGraph.draw(o)
