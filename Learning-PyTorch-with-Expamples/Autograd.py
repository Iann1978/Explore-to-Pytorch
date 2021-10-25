import torch
import math

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)


a = torch.randn((), requires_grad=True)
b = torch.randn((), requires_grad=True)
c = torch.randn((), requires_grad=True)
d = torch.randn((), requires_grad=True)

lr = 1e-6
for t in range(2000):

    # Forward pass
    y_pred = a + b*x + c*x**2 + d*x**3
    # loss = torch.square(y_pred - y).sum()
    loss = (y_pred - y).pow(2).sum()

    if t%100==99:
        print(t, loss)

    loss.backward()

    # grad_y_pred = 2 * (y_pred - y)
    # grad_a = grad_y_pred.sum()
    # grad_b = (grad_y_pred*x).sum()
    # grad_c = (grad_y_pred*x**2).sum()
    # grad_d = (grad_y_pred*x**3).sum()

    # a -= lr * grad_a
    # b -= lr * grad_b
    # c -= lr * grad_c
    # d -= lr * grad_d

    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad
        c -= lr * c.grad
        d -= lr * d.grad

        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: {a} + {b}x + {c}xx + {d}xxx', a, b,c,d)