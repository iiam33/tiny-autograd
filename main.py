from number import Number
from graphviz import Graph
from pathlib import Path

a = Number(data=2.0, label='a')
b = Number(data=-3.0, label='b')
c = Number(data=10.0, label='c')
d = a * b
d.label = 'd'
e = d + c
e.label = 'e'
f = Number(data=-2.0, label='f')
L = e * f
L.label = 'L'

print('a', a)
print('b', b)
print('c', c)
print('d', d)
print('e', e)
print('f', f)
print('L', L)

# graph = Graph(format='jpeg', graph_attr={'rankdir': 'LR'})
# graph.node(a.label, "%.4f" % a.data)
# graph.node(b.label, "%.4f" % b.data)
# graph.node(c.label, "%.4f" % c.data)
# graph.edges(['ac', 'bc'])
# graph.edge('a', 'c')
# graph.edge('b', 'c')

pwd = Path.cwd()
dir = Path.joinpath(pwd, 'graph', 'graph.gv')


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
            graph.node(name=str(id(node)), label="{%s | %.4f}" % (node.label, node.data), shape='record')

        for head, tail in edges:
            graph.edge(str(id(head)), str(id(tail)))

        graph.render(dir)


CustomGraph.draw(L)
