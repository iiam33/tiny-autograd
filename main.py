from number import Number
from pathlib import Path
from customgraph import CustomGraph

pwd = Path.cwd()
dir = Path.joinpath(pwd, 'graph', 'graph_tenh.gv')


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

# o = n.tanh()
e = (2*n).exp()
e.label = 'e'
o = (e - 1) / (e + 1)
o.label = 'o'

o.backward()

CustomGraph.draw(o)
