from main import CustomGraph
from pathlib import Path
from nnetwork import MLP

data = [2.0, 4.0, -1.2, 3.1]
n = MLP(4, [3, 3, 1])

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
# desired output
ys = [1.0, -1.0, -1.0, 1.0]


pwd = Path.cwd()

for i in range(300):
    # forward pass
    ypred = [n(x) for x in xs]
    # implementing mean square error loss function with y_ground_truth and y_output
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # zero grad
    for p in n.parameters():
        p.grad = 0.0

    # backward pass
    loss.backward()

    for p in n.parameters():
        p.data += -0.15 * p.grad

    print(i, loss.data)

print(ypred)
