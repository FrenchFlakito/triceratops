class Network():
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to the network
    def add_layer(self, layer):
        self.layers.append(layer)
    
    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        sample_size = len(input_data)
        result = []
        for i in range(sample_size):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        sample_size = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(sample_size):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)   # forward propagation

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
