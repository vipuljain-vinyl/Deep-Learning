import torch

# This is your BaseUnit class. You will inherit from this class to
# create custom layers.
class BaseUnit:
    def __init__(self, lr):
        self.eval_mode = False
        self.lr = lr

    def eval(self):
        self.eval_mode = True

    def train(self):
        self.eval_mode = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(BaseUnit):
    def __init__(self, d_in, d_out, lr=1e-3):
        super().__init__(lr)
        # create the parameter W and initialize it from a normal
        # distribution with mean 0 and std 0.05. Check torch.randn
        # for this.
        self.W = torch.randn(d_in, d_out) * 0.05
        # create the parameter b and initialize it to zeros
        self.b = torch.zeros(d_out)
        self.d_in = d_in
        self.d_out = d_out
        # self.grad_comps for each parameter
        self.h_W = None
        self.h_b = None

    def forward(self, X):
        # X is a batch of data of shape n x d_in
        # print()
        # print(X)
        # print(X.shape[0])
        n = X.shape[0]
        # calculate out = X @ W + b. Remember to reshape b so that it
        # adds elementwise to each row.
        self.X = X.clone()
        out = X @ self.W + self.b.unsqueeze(0)

        if not self.eval_mode:
            # You are in training mode.
            # Compute self.h_W = d(out)/d(W) and self.h_b = d(out)/d(b).
            # Remember to preserve the batch dimension as it is
            # collapsed only during the final gradient computation
            if self.h_W is None:
                self.h_W = X.unsqueeze(2).repeat(1, 1, self.d_out)
                self.h_b = torch.ones(n, self.d_out)

        return out

    def backward(self, grad):
        # grad is of shape n x d_out
        n = grad.shape[0]
        # Create placeholders for the gradients of W and b
        grad_W = []
        grad_b = []

        # Calculate the gradients for W and b. Use a for loop in the
        # beginning to ensure the correctness of your implementation
        for i in range(n):
            # For sample i, d(out_i)/dW = outer product of X[i] and grad[i]
            grad_W.append(torch.ger(self.X[i], grad[i]))
            # For sample i, d(out_i)/db = grad[i]
            grad_b.append(grad[i])
        
        # Average the gradients over the batch dimension
        grad_W = torch.stack(grad_W, dim=0).mean(0).mean(0)
        grad_b = torch.stack(grad_b, dim=0).mean(0).mean(0)

        # Compute grad_for_next BEFORE updating weights.
        # Since out = X @ W + b, d(out)/dX = W^T, so:
        grad_for_next = grad @ self.W.t()

        # Update the parameters using the gradients
        self.W = self.W - self.lr * grad_W
        self.b = self.b - self.lr * grad_b

        # Return the grad for the previous layer

        return grad_for_next

class ReLU(BaseUnit):
    def __init__(self, lr=None):
        super().__init__(lr)
        self.sign = None

    def forward(self, X):
        if not self.eval_mode:
            # store the information required for the backward pass
            self.mask = (X > 0).float()
        
        # Compute the ReLU activation
        out = torch.relu(X)
        return out

    def backward(self, grad):
        # There is no gradient for ReLU since there are no parameters.
        # However, you must compute the gradient for the previous layer
        grad_for_next = grad * self.mask

        return grad_for_next

class MSE(BaseUnit):
    def __init__(self, lr=None):
        super().__init__(lr)
        self.grad_return = None

    def forward(self, yhat, y):
        if not self.eval_mode:
            # store the parts required for the backward pass
            self.diff = yhat - y
        
        # Calculate the mean squared error
        error = torch.mean((yhat - y) ** 2)
        return error

    def backward(self, grad=None):
        # There is no gradient for MSE since there are no parameters.
        # Return the gradient for the previous layer
        n = self.diff.numel()
        grad_for_next = (2.0 / n) * self.diff
        
        return grad_for_next