import random
from utils.engine import Value
class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        out = None
        ####################################################################
        # TODO: Implement the Layer forward pass.                      #
        ####################################################################
        # Replace "pass" statement with your code
        
        dummy_output_1 = Value(0)
        for wi, xi in zip(self.w, x):
            dummy_output_1 += wi * xi
        dummy_output_1 += self.b
        
        if self.nonlin:
            out = dummy_output_1.relu()
            
        else:
            out = dummy_output_1
            
        ####################################################################
        #                         END OF YOUR CODE                         #
        ####################################################################
        
        return out


    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = None
        ####################################################################
        # TODO: Implement the Layer forward pass.                      #
        ####################################################################
        # Replace "pass" statement with your code
        
        out = [neuron(x) for neuron in self.neurons]
        
        #####################################################################
        #                          END OF YOUR CODE                         #
        #####################################################################
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        out = None
        ####################################################################
        # TODO: Implement the MLP forward pass.                            #
        ####################################################################
        # Replace "pass" statement with your code
        for layer in self.layers:
            x  = layer(x)
        out = x
        #####################################################################
        #                          END OF YOUR CODE                         #
        #####################################################################
        return out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
