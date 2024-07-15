import argparse

from value import Value
from nn import MLP
from vis_backprop import draw_graph



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=int, default=3, help="Select which example computational graph should be shown.")
    args = parser.parse_args()

    return args


def main():
    """
    This script implements four different examples of forward and backward pass through compuational graphs.
    """

    args = get_args()

    # example 1: L = (a*b + c) * f
    if args.example == 0:

        # define inputs
        a = Value(2.0, label="a")
        b = Value(-3.0, label="b")
        c = Value(10.0, label="c")
        f = Value(-2.0, label="f")

        # forward pass
        e = a*b 
        e.label = "e"
        d = e + c 
        d.label = "d"
        L = d * f
        L.label = "L"

        # backward pass
        L.backward()

        # show the computational graph
        graph = draw_graph(L)
        graph.view()
    

    # example 2: y = x1*W1 + x2*W2 + b
    elif args.example == 1:
    
        # define inputs
        x1 = Value(2.0, label="x1")
        x2 = Value(0.0, label="x2")
        w1 = Value(-3.0, label="w1")
        w2 = Value(1.0, label="w2")
        b = Value(6.8813735870195432, label="b")

        # forward pass
        x1w1 = x1*w1
        x1w1.label = "x1w1"
        x2w2 = x2*w2
        x2w2.label = "x2w2"
        x1w1x2w2 = x1w1+x2w2
        x1w1x2w2.label = "x1w1x2w2"
        n = x1w1x2w2 + b
        n.label="n"
        o = n.tanh()

        # backward pass
        o.backward()

        # show the computational graph
        graph = draw_graph(o)
        graph.view()


    # example 3: MLP with single input
    elif args.example == 2:

        # define inputs and weights
        x = [2.0, 3.0, -1.0]
        n = MLP(3, [4,4,1])

        # forward pass
        out = n(x)

        # backward pass
        out.backward()

        # show the computational graph
        graph = draw_graph(out)
        graph.view()
    

    # example 4: MLP with multiple inputs and loss function
    elif args.example == 3:

        # define inputs, labels and weights
        xs = [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            ]
        
        y = [1.0, -1.0, -1.0, 1.0]
        ys = [Value(_) for _ in y]

        n = MLP(3, [4,4,1])

        # forward pass
        out = [n(x) for x in xs]

        # calculate loss (mean squared error)
        loss = sum((yout.data - ygt)**2 for ygt, yout in zip(ys, out))
        
        # backward
        loss.backward()

        # show the computational graph
        graph = draw_graph(loss)
        graph.view()



if __name__ == "__main__":
    main()