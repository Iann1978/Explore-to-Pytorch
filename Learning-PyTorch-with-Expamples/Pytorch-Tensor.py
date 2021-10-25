import torch
import math
import torch

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

a = torch.randn(())
b = torch.randn(())
c = torch.randn(())
d = torch.randn(())

learning_rate = 1e-6
for t in range(2000):
    # Forward pass
    y_pred = a + b*x + c*x**2 + d*x**3

    # Compute and print loss
    loss = torch.square(y_pred - y).sum()
    if t%100==99:
        print(t,loss)

    grad_y_pred = 2.0*(y_pred-y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred*x).sum()
    grad_c = (grad_y_pred*x*x).sum()
    grad_d = (grad_y_pred*x*x*x).sum()

    a -= learning_rate*grad_a
    b -= learning_rate*grad_b
    c -= learning_rate*grad_c
    d -= learning_rate*grad_d

print(f'Result:y={a}={b}x+{c}xx+{d}xxx')

