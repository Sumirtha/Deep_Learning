import numpy as np

class VanillaRNN:

    def __init__(self, vocab_size, hidden_size=72, learning_rate=0.05):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lr = learning_rate

        # Input to Hidden
        self.U = np.random.randn(hidden_size, vocab_size) * 0.01

        # Hidden to Hidden
        self.W = np.random.randn(hidden_size, hidden_size) * 0.01

        # Hidden to Output
        self.V = np.random.randn(vocab_size, hidden_size) * 0.01

        # Biases
        self.b = np.zeros((hidden_size, 1))
        self.c = np.zeros((vocab_size, 1))

    # SOFTMAX

    def softmax(self, x):

        # Numerical stability
        x = x - np.max(x)

        exp_x = np.exp(x)

        return exp_x / np.sum(exp_x)

    # FORWARD PASS

    def forward(self, inputs, targets):

        """
        inputs  : list of integer character indices
        targets : list of integer target indices
        """

        # Store values needed during BPTT
        xs = {}
        hs = {}
        os = {}
        ps = {}

        # Initial hidden state
        hs[-1] = np.zeros((self.hidden_size, 1))

        loss = 0

        # Process each time step
        for t in range(len(inputs)):

            # One-hot encode input

            x = np.zeros((self.vocab_size, 1))
            x[inputs[t]] = 1

            xs[t] = x

            # Hidden state

            hs[t] = np.tanh(
                np.dot(self.U, xs[t])
                + np.dot(self.W, hs[t - 1])
                + self.b
            )

            # Output logits

            os[t] = (
                np.dot(self.V, hs[t])
                + self.c)

                    # Probability

            ps[t] = self.softmax(os[t])

            # Cross entropy loss

            loss += -np.log(ps[t][targets[t], 0] + 1e-12)

        return loss, xs, hs, os, ps


    # BACKPROPAGATION THROUGH TIME

    def backward(self, inputs, targets, xs, hs, os, ps):
        # Initialize gradients

        dU = np.zeros_like(self.U)
        dW = np.zeros_like(self.W)
        dV = np.zeros_like(self.V)
        db = np.zeros_like(self.b)
        dc = np.zeros_like(self.c)

        # Gradient flowing backward through hidden states
        dh_next = np.zeros_like(hs[0])

        # Go backwards through time
        for t in reversed(range(len(inputs))):

            # Gradient of softmax + cross entropy

            dy = ps[t].copy()
            dy[targets[t]] -= 1

            # Output layer gradients

            dV += np.dot(dy, hs[t].T)
            dc += dy

            # Gradient flowing into hidden state
            dh = np.dot(self.V.T, dy) + dh_next

            # tanh derivative

            dtanh = (
                1 - hs[t] ** 2
            ) * dh

            # Input to hidden

            dU += np.dot(dtanh, xs[t].T)

            # Hidden to hidden

            dW += np.dot(
                dtanh,
                hs[t - 1].T)

            # Hidden bias

            db += dtanh

            # Send gradient to previous timestep

            dh_next = np.dot(
                self.W.T,
                dtanh)

               # GRADIENT CLIPPING

        for gradient in [dU, dW, dV, db, dc]:

            np.clip(
                gradient,
                -5,
                5,
                out=gradient)


        # UPDATE PARAMETERS

        self.U -= self.lr * dU
        self.W -= self.lr * dW
        self.V -= self.lr * dV
        self.b -= self.lr * db
        self.c -= self.lr * dc

    # TRAIN

    def train(self, sequences, epochs=1000):

        for epoch in range(epochs):

            total_loss = 0

            for inputs, targets in sequences:

                loss, xs, hs, os, ps = self.forward(
                    inputs,
                    targets)

                self.backward(
                    inputs,
                    targets,
                    xs,
                    hs,
                    os,
                    ps)

                total_loss += loss

            if epoch % 100 == 0:

                print(f"Epoch {epoch:4d} | "f"Loss = {total_loss:.4f}")

   # PREDICTION

    def predict(self, inputs):

        h = np.zeros(
            (self.hidden_size, 1))

        predictions = []

        for index in inputs:

            # One-hot input
            x = np.zeros(
                (self.vocab_size, 1))

            x[index] = 1

            # Hidden state
            h = np.tanh(
                np.dot(self.U, x)
                + np.dot(self.W, h)
                + self.b)

            # Output
            o = (
                np.dot(self.V, h)
                + self.c)

            # Probability
            p = self.softmax(o)

            prediction = np.argmax(p)

            predictions.append(prediction)

        return predictions

# 2. DATA
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

char_to_index = {
    char: i
    for i, char in enumerate(alphabet)}

index_to_char = {
    i: char
    for i, char in enumerate(alphabet)}

# CREATE TRAINING SEQUENCES

sequences = []

sequence_length = 5

for i in range(len(alphabet) - sequence_length):

    input_string = alphabet[i:i + sequence_length]

    target_string = alphabet[
        i + 1:i + sequence_length + 1
    ]

    inputs = [
        char_to_index[c]
        for c in input_string]

    targets = [
        char_to_index[c]
        for c in target_string]

    sequences.append(
        (inputs, targets))

# 3. CREATE MODEL
rnn = VanillaRNN(
    vocab_size=26,
    hidden_size=72,
    learning_rate=0.05)

# 4. TRAIN
rnn.train(
    sequences,
    epochs=5000)

# 5. TEST
test_sequence = "ABCDE"

test_inputs = [
    char_to_index[c]
    for c in test_sequence]

predictions = rnn.predict(
    test_inputs)

predicted_string = "".join(
    index_to_char[i]
    for i in predictions)

print("\nInput:     ", test_sequence)
print("Prediction:", predicted_string)