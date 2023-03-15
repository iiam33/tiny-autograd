from main import CustomGraph
from pathlib import Path
from nnetwork import MLP

data = [2.0, 4.0, -1.2, 3.1]
# n = Neuron(1)
n = MLP(4, [3, 3, 1])
# n(data)

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
# desired output
ys = [1.0, -1.0, -1.0, 1.0]


pwd = Path.cwd()

ypred = [n(x) for x in xs]

print(ypred)

# for y in range(len(ypred)):
#     dir = Path.joinpath(pwd, 'graph', f'graph_nn{y}.gv')
#     CustomGraph.draw(root=n(data), path=dir)

# implementing mean square error loss function with y_ground_truth and y_output
print("lost function", [(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
lost = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

print('lost', lost)
