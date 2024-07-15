import math



class Value:
    """
    This class takes a scalar value of a computational graph and keeps track of it.
    It allows the calculation of the forward as well as the backward pass.

    Params:
        - data: This is a scalar value of a node within the computation graph (int)
        - _children: This set contains the child nodes of the current node (which nodes were involved in the calculation of the value of the current node?) (tuple)
        - _op: This is a sign of the operator which was used for the calculation (str)
        - label: This is the label of the node (str)
    """
    
    def __init__(self, data, _children=(), _op="", label=""):

        self.data = data
        self.grad = 0.0
        self._backward = lambda:None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        """
        Returns the sum of the data of two Value objects.
        """

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        """
        Returns the product of the data of two Value objects.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        """
        Returns the exponentiation of two Value objects.
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        """
        Return the relu() of the data of a Value object.
        """
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        """
        Returns the tanh() of the data of a Value object.
        """
        x = self.data
        t = (math.exp(2*x) - 1)/ (math.exp(2*x) + 1)
        out = Value(t, (self, ), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        """
        Returns the exp() of the data of a Value object.
        """
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        """
        Runs backpropagation through all nodes of the computational graph starting from the last node.
        """

        topo = []
        visited = set()

        def build_topo(v):
            
            if v not in visited:
                visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
        
        build_topo(self)

        self.grad = 1
        for node in reversed(topo):
            node._backward()
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self *other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
