import numpy as np


class RNN:

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: list):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.sizes = [self.input_dim] + self.hidden_layers + [self.output_dim]
        # kaiming-init weights and biases
        self.Wb = [[np.random.normal(0., np.sqrt(2. / n), size=(m, n)), np.zeros((m, 1))] for m, n in
                   zip(self.sizes[1:], self.sizes[:-1])]
        sz = self.sizes[-1]
        # add temporal layer
        self.Wb.append([np.random.normal(0., np.sqrt(2. / sz), size=(sz, sz)), np.zeros((2, 1))])
        self.grad = [[np.empty(W.shape), np.empty(b.shape)] for W, b in self.Wb]
        self.BLANK = 0  # symbol used for blank

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """ activation function """
        return 1. / (1. + np.exp(-x))

    def d_sigmoid(self, x: np.ndarray) -> np.ndarray:
        """ simplified derivative of activation function """
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def ctc_loss(self, y: np.ndarray, x_z: np.ndarray) -> tuple[float, np.ndarray]:
        """
        The CTC loss function

        :param y: sequence of network outputs (probability distributions)
        :param x_z: sequence (x,z) from the set of training examples S
        :param self.BLANK: the blank symbol
        :return: negative log likelihood and gradients
        """

        def forward(y: np.ndarray, x_z: np.ndarray, L: int, T: int) -> tuple[np.ndarray, float]:
            """
            Forward iteration through the graph.

            :param T: time steps
            :param L: length of labellings extended with blanks
            :param y: sequence of network outputs (probability distributions)
            :param x_z: sequence (x,z) from the set of training examples S

            Initialization of the forward variable (alpha):
            a_t(s) = 0 ∀s < 1
            a_t(s) = 0 ∀s < |l′| − 2(T − t) − 1
            """
            # initialize alpha
            a = np.zeros((L, T))
            # initialize starting points on graph
            a[0, 0] = y[self.BLANK, 0]
            a[1, 0] = y[x_z[0], 0]
            # log likelihood for forward variable
            ll_forward = 0.0
            # loop through time
            for t in range(1, T):
                first = max(0, L - (2 * (T - t)))
                last = min((2 * t) + 2, L)
                # loop over symbols
                for s in range(first, L):
                    l = (s - 1) // 2
                    # s is indexing a blank (extended sequence has alternating blanks)
                    if s % 2 == 0:
                        if s == 0:
                            a[s, t] = a[s, t - 1] * y[self.BLANK, t]
                        else:
                            a[s, t] = (a[s, t - 1] + a[s - 1, t - 1]) * y[self.BLANK, t]
                    # repeated label
                    elif (s == 1) or (x_z[l] == x_z[l - 1]):
                        a[s, t] = (a[s, t - 1] + a[s - 1, t - 1]) * y[x_z[l], t]
                    else:
                        a[s, t] = (a[s, t - 1] + a[s - 1, t - 1] + a[s - 2, t - 1]) * y[x_z[l], t]
                # rescale forward variable
                C = np.sum(a[first:last, t])
                a[first:last, t] /= C
                ll_forward += np.log(C)
            return a, ll_forward

        def backward(y: np.ndarray, x_z: np.ndarray, L: int, T: int) -> np.ndarray:
            """
            Backward iteration through the graph.

            :param T: time steps
            :param L: length of labellings extended with blanks
            :param y: sequence of network outputs (probability distributions)
            :param x_z: sequence (x,z) from the set of training examples S

            Initialization of the backward variable (beta):
            b_t(s) = 0 ∀s > 2t
            b_t(s) = 0 ∀s > |l′|
            """
            # initialize beta
            b = np.zeros((L, T))
            # initialize starting points on graph
            b[-1, -1] = y[self.BLANK, -1]
            b[-2, -1] = y[x_z[-1], -1]
            for t in range(T - 2, -1, -1):
                first = max(0, L - (2 * (T - t)))
                last = min((2 * t) + 2, L)
                for s in range(last - 1, -1, -1):
                    l = (s - 1) // 2
                    # s is indexing a blank (extended sequence has alternating blanks)
                    if s % 2 == 0:
                        if s == L - 1:
                            b[s, t] = b[s, t + 1] * y[self.BLANK, t]
                        else:
                            b[s, t] = (b[s, t + 1] + b[s + 1, t + 1]) * y[self.BLANK, t]
                    # repeated label
                    elif (s == L - 2) or (x_z[l] == x_z[l + 1]):
                        b[s, t] = (b[s, t + 1] + b[s + 1, t + 1]) * y[x_z[l], t]
                    else:
                        b[s, t] = (b[s, t + 1] + b[s + 1, t + 1] + b[s + 2, t + 1]) * y[x_z[l], t]
                # rescale backward variable
                b[first:last, t] /= np.sum(b[first:last, t])
            return b

        L = 2 * x_z.shape[0] + 1
        T = y.shape[1]
        # forward backward
        a, ll_forward = forward(y, x_z, L, T)
        b = backward(y, x_z, L, T)
        # zero grads
        grad = np.zeros(y.shape)
        # total probability of the paths
        gamma = a * b
        for s in range(L):
            # blank
            if s % 2 == 0:
                grad[self.BLANK, ...] += gamma[s, ...]
                gamma[s, ...] /= y[self.BLANK, ...]
            else:
                grad[x_z[(s - 1) // 2], ...] += gamma[s, ...]
                gamma[s, ...] /= (y[x_z[(s - 1) // 2], ...])
        grad = y - (grad / (y * np.sum(gamma, axis=0)))
        return -ll_forward, grad

    def forward(self, X: np.ndarray, Z: np.ndarray, T: int, len_Wb: int) -> tuple[float, np.ndarray]:
        """
        Forward propagation

        :param X: (input space) set of all sequences of m-dimensional vectors
        :param Z: (target space) set of all sequences over the alphabet L of labels
        :param T: time steps
        :param len_Wb: length of the list of weights and biases
        """
        self.A = [np.zeros((s, T)) for s in self.sizes]  # zero activations
        self.A[0] = X
        i = 1
        for l in range(len_Wb - 1):
            W, b = self.Wb[l]
            self.A[i] = W @ self.A[i - 1] + b
            if l == -2:
                # loop over time t for recurrent layer
                for t in range(T):
                    if t > 0:
                        self.A[i][..., t] += self.Wb[-1][0] @ self.A[i][..., t - 1]
                    if i <= len_Wb - 2:
                        self.A[i][..., t] = self.sigmoid(self.A[i][..., t])
            elif i <= len_Wb - 2:
                self.A[i] = self.sigmoid(self.A[i])  # hidden layer activation
            i += 1
        # get probs from last activation
        probs = np.exp(self.A[-1] - np.max(self.A[-1], axis=0))
        probs /= np.sum(probs, axis=0)
        # pass probs and target_space to ctc_loss
        loss, delta = self.ctc_loss(probs, Z.squeeze())
        return loss, delta

    def backward(self, T: int, delta_fwd_pass: np.ndarray, len_Wb: int, lr: float) -> None:
        """
        Backprop

        :param T: time steps
        :param delta_fwd_pass: delta computed from the forward pass
        :param len_Wb: length of the list of weights and biases
        :param lr: learning rate
        """
        # zero gradients
        self.grad = [[np.zeros(w.shape), np.zeros(b.shape)] for w, b in self.Wb]
        # BPTT
        for t in reversed(range(T)):
            # use delta computed by the forward pass
            delta = delta_fwd_pass[..., t].T
            # calculate partials for output layer
            self.grad[len_Wb - 2][0] += delta.reshape(-1, 1) @ self.A[-2][..., t].reshape(-1, 1).T
            self.grad[len_Wb - 2][1] += delta.reshape(-1, 1)
            delta = self.Wb[len_Wb - 2][0].T @ delta
            # backprop delta through the rest of the network
            for i in range(len(self.hidden_layers) - 1, -1, -1):
                # calculate partials
                delta = delta * self.d_sigmoid(self.A[i + 1][..., t])
                self.grad[i][0] += delta.reshape(-1, 1) @ self.A[i][..., t].T.reshape(1, -1)
                self.grad[i][1] += delta.reshape(-1, 1)
                W, b = self.Wb[i]
                # update delta
                delta = W.T @ delta
        # update RNN weights and biases
        self.Wb = [[wb[0] - lr * g[0], wb[1] - lr * g[1]] for wb, g in zip(self.Wb, self.grad)]

    def train(self, X: np.ndarray, Z: np.ndarray, lr: float = 1e-2, epochs: int = 10) -> None:
        """
        The training loop

        :param X: (input space) the set of all sequences of m dimensional real-valued vectors, each of length T
        :param Z: (target space) the set of all sequences over the (finite) alphabet L of labels
                  (a.k.a labellings or label sequences), each of length U
        :param lr: learning rate
        :param epochs: number of iterations over the pairs of training examples
        """
        for i in range(epochs):
            T = X.shape[1]
            # calculate loss and delta from forward pass
            loss, delta_output = self.forward(X, Z, T, len(self.Wb))
            # back propagate delta
            self.backward(T, delta_output, len(self.Wb), lr)
            print(f'{loss = }')


def main():
    np.random.seed(3)
    d_in = 3
    n_symbols = 6
    d_out = n_symbols + 1
    len_input_seq = 5
    len_target_seq = 3
    hidden_layers = [64, 32, 16]

    # input space
    X = np.random.randn(d_in, len_input_seq)
    # target space
    Z = np.random.randint(low=0, high=d_out, size=(len_target_seq, 1))

    nn = RNN(d_in, d_out, hidden_layers)
    nn.train(X, Z, lr=1e-2, epochs=10)


if __name__ == '__main__':
    main()
