from graphviz import Graph


class CustomGraph:
    def create(root):
        nodes, edges = set(), set()

        def build(root):
            if root not in nodes:
                nodes.add(root)
            for node in root._prev:
                nodes.add(node)
                edges.add((node, root))
                build(node)

        build(root)
        return nodes, edges

    def draw(root, path):
        nodes, edges = CustomGraph.create(root)

        graph = Graph(format='jpeg', graph_attr={'rankdir': 'LR'})

        for node in nodes:
            graph.node(name=str(id(node)), label="{%s | data: %.4f | grad: %.4f}" % (
                node.label, node.data, node.grad), shape='record')

        for head, tail in edges:
            graph.edge(str(id(head)), str(id(tail)))

        graph.render(path)
