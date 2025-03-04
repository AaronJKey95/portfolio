import numpy as np
from initialise import simple_classifier_data, complex_classifier_data, make_chart, clear_folder, deep_copy_dict


def ReLU(x):
    return np.maximum(x, 0)


# loss function
def mean_squared_error(y, y_pred):
    sq_res = np.square(y_pred - y)
    mse = np.mean(sq_res)
    return mse


def node(input, weight, bias, activation_func):
    for i in range(input.shape[1]):
        input[:, i] = input[:, i] * weight[i]
    input = input.sum(axis=1)
    input = input + bias
    output = activation_func(input)
    return output


def hidden_layer(input, number_nodes, activation_func, weights=None, bias=None):
    if not weights:
        weights = {}
        for i in range(number_nodes):
            weights.update({i: {}})
            for j in range(input.shape[1]):
                weights[i].update({j: np.random.uniform(-1, 1)})

    if not bias:
        bias = {}
        for i in range(number_nodes):
            bias.update({i: np.random.uniform(-1, 1)})

    output = {}
    for i in range(number_nodes):
        output.update({i: node(input, weights[i], bias[i], activation_func)})

    output = list(output.values())
    output = [arr.reshape(-1, 1) for arr in output]
    output = np.concatenate(output, axis=1)
    return output, weights, bias


def output_layer(input, weights=None, bias=None):
    if not weights:
        weights = []
        for i in range(input.shape[1]):
            weights.append(np.random.uniform(-1, 1))
    if not bias:
        bias = np.random.uniform(-1, 1)
    adj_data = input.copy()
    for i in range(input.shape[1]):
        adj_data[:, i] = adj_data[:, i] * weights[i]

    adj_data = adj_data.sum(axis=1)
    data = np.clip(adj_data + bias, -500, 500)
    sigmoid = 1 / (1 + np.exp(-data))
    return sigmoid, weights, bias


class neural_network:
    def __init__(self, x, y):
        self.input_layer = x
        self.target = y
        self.sample_target = None
        self.data = x.copy()
        self.output_layer = None
        self.output_weights = None
        self.output_bias = None
        self.weights_bias = [1, 0]
        self.hidden_layers = 0
        self.hidden_layer_weights = {}
        self.hidden_layer_bias = {}
        self.hidden_layer_function = {}
        self.hidden_layer_description = "No Hidden Layers"

    def add_hidden_layer(self, number_nodes, activation_func):

        self.data, hidden_layer_weights, hidden_layer_bias = hidden_layer(self.data, number_nodes, activation_func)
        self.hidden_layer_weights.update({self.hidden_layers: hidden_layer_weights})
        self.hidden_layer_bias.update({self.hidden_layers: hidden_layer_bias})
        self.hidden_layer_function.update({self.hidden_layers: activation_func})
        self.hidden_layers += 1
        if self.hidden_layers == 1:
            self.hidden_layer_description = f"Hidden Layer {self.hidden_layers} shape: {self.data.shape} \n" \
                                            f"Hidden Layer {self.hidden_layers} Weights: {self.hidden_layer_weights[self.hidden_layers - 1]} \n" \
                                            f"Hidden Layer {self.hidden_layers} Bias: {self.hidden_layer_bias[self.hidden_layers - 1]} \n" \
                                            f"Hidden Layer {self.hidden_layers} Function: {activation_func.__name__} \n\n"
        else:
            self.hidden_layer_description = self.hidden_layer_description + \
                                            f"Hidden Layer {self.hidden_layers} shape: {self.data.shape} \n" \
                                            f"Hidden Layer {self.hidden_layers} Weights: {self.hidden_layer_weights[self.hidden_layers - 1]} \n" \
                                            f"Hidden Layer {self.hidden_layers} Bias: {self.hidden_layer_bias[self.hidden_layers - 1]} \n" \
                                            f"Hidden Layer {self.hidden_layers} Function: {activation_func.__name__} \n\n"

    def architecture(self):
        self.weights_bias = [0, 0]
        self.weights_bias[0] += len(self.output_weights)
        self.weights_bias[1] += 1
        for i in range(self.hidden_layers):
            for j in range(len(self.hidden_layer_weights[i])):
                for k in range(len(self.hidden_layer_weights[i][j])):
                    self.weights_bias[0] += 1
            for j in range(len(self.hidden_layer_bias[i])):
                self.weights_bias[1] += 1
        print(
            f"======================================================================================================\n"
            f"Input shape: {self.input_layer.shape} \n\n"
            f"{self.hidden_layer_description}"
            f"Output shape: {self.output_layer.shape} \n"
            f"Output Weights: {self.output_weights} \n"
            f"Output Bias: {self.output_bias} \n"
            f"Total Count of Weights and Bias: {self.weights_bias} \n"
            f"Output Error: {mean_squared_error(self.output_layer, self.target)} \n")

    def add_output_layer(self):
        self.output_layer, self.output_weights, self.output_bias = output_layer(self.data)

    def run_with_set_parameters(self,
                                chart_data=None,
                                adj_output_weights=None,
                                adj_output_bias=None,
                                adj_hidden_weights=None,
                                adj_hidden_bias=None):
        if chart_data is None:
            self.data = self.input_layer.copy()
            self.sample_target = self.target.copy()
        else:
            self.data = chart_data
        for i in range(self.hidden_layers):
            if adj_hidden_weights:
                self.data, w, b = \
                    hidden_layer(self.data,
                                 max(self.hidden_layer_bias[i].keys()) + 1,
                                 self.hidden_layer_function[i],
                                 adj_hidden_weights[i],
                                 self.hidden_layer_bias[i])
            elif adj_hidden_bias:
                self.data, w, b = \
                    hidden_layer(self.data,
                                 max(self.hidden_layer_bias[i].keys()) + 1,
                                 self.hidden_layer_function[i],
                                 self.hidden_layer_weights[i],
                                 adj_hidden_bias[i])
            else:
                self.data, w, b = \
                    hidden_layer(self.data,
                                 max(self.hidden_layer_bias[i].keys()) + 1,
                                 self.hidden_layer_function[i],
                                 self.hidden_layer_weights[i],
                                 self.hidden_layer_bias[i])
        if adj_output_weights:
            self.output_layer, w, b = output_layer(self.data, adj_output_weights, self.output_bias)
        elif adj_output_bias:
            self.output_layer, w, b = output_layer(self.data, self.output_weights, adj_output_bias)
        else:
            self.output_layer, w, b = output_layer(self.data, self.output_weights, self.output_bias)
        return self.output_layer

    def make_chart(self, i, first_chart=False):
        make_chart(self.input_layer,
                   self.target,
                   "MLP",
                   self.run_with_set_parameters,
                   error=mean_squared_error(self.sample_target, self.output_layer),
                   i=i,
                   first_chart=first_chart)

    def back_prop(self, learning_rate=0.01):
        d = 1e-9
        self.run_with_set_parameters()

        for i in range(len(self.output_weights)):
            weights_d = self.output_weights.copy()
            weights_d[i] = weights_d[i] + d
            loss_1 = mean_squared_error(self.sample_target, self.run_with_set_parameters())
            loss_2 = mean_squared_error(self.sample_target, self.run_with_set_parameters(adj_output_weights=weights_d))
            d_loss_tan = (loss_2 - loss_1) / d
            weights_d = self.output_weights.copy()
            weights_d[i] = weights_d[i] - d_loss_tan * learning_rate
            loss_2 = mean_squared_error(self.sample_target, self.run_with_set_parameters(adj_output_weights=weights_d))
            if loss_1 > loss_2:
                self.output_weights = weights_d

        bias_d = self.output_bias
        bias_d = bias_d + d
        loss_1 = mean_squared_error(self.sample_target, self.run_with_set_parameters())
        loss_2 = mean_squared_error(self.sample_target, self.run_with_set_parameters(adj_output_bias=bias_d))
        d_loss_tan = (loss_2 - loss_1) / d
        bias_d = self.output_bias - d_loss_tan * learning_rate
        loss_2 = mean_squared_error(self.sample_target, self.run_with_set_parameters(adj_output_bias=bias_d))
        if loss_1 > loss_2:
            self.output_bias = bias_d

        for i in range(self.hidden_layers):
            for j in range(len(self.hidden_layer_weights[i])):
                for k in range(len(self.hidden_layer_weights[i][j])):
                    self.weights_bias[0] += 1
                    weights_d = deep_copy_dict(self.hidden_layer_weights)
                    weights_d[i][j][k] = weights_d[i][j][k] + d
                    loss_1 = mean_squared_error(self.sample_target, self.run_with_set_parameters())
                    loss_2 = mean_squared_error(self.sample_target, self.run_with_set_parameters(adj_hidden_weights=weights_d))
                    d_loss_tan = (loss_2 - loss_1) / d
                    weights_d = deep_copy_dict(self.hidden_layer_weights)
                    weights_d[i][j][k] = weights_d[i][j][k] - d_loss_tan * learning_rate
                    loss_2 = mean_squared_error(self.sample_target, self.run_with_set_parameters(adj_hidden_weights=weights_d))
                    if loss_1 > loss_2:
                        self.hidden_layer_weights = weights_d

            for j in range(len(self.hidden_layer_bias[i])):
                bias_d = deep_copy_dict(self.hidden_layer_bias)
                bias_d[i][j] = bias_d[i][j] + d
                loss_1 = mean_squared_error(self.sample_target, self.run_with_set_parameters())
                loss_2 = mean_squared_error(self.sample_target, self.run_with_set_parameters(adj_hidden_bias=bias_d))
                d_loss_tan = (loss_2 - loss_1) / d
                bias_d = deep_copy_dict(self.hidden_layer_bias)
                bias_d[i][j] = bias_d[i][j] - d_loss_tan * learning_rate
                loss_2 = mean_squared_error(self.sample_target, self.run_with_set_parameters(adj_hidden_bias=bias_d))
                if loss_1 > loss_2:
                    self.hidden_layer_bias = bias_d

        for i in range(self.hidden_layers):
            if i == 0:
                self.hidden_layer_description = f"Hidden Layer {i+1} shape: {self.data.shape} \n" \
                                                f"Hidden Layer {i+1} Weights: {self.hidden_layer_weights[self.hidden_layers - 1]} \n" \
                                                f"Hidden Layer {i+1} Bias: {self.hidden_layer_bias[self.hidden_layers - 1]} \n" \
                                                f"Hidden Layer {i+1} Function: {self.hidden_layer_function[i].__name__} \n\n"
            else:
                self.hidden_layer_description = self.hidden_layer_description + \
                                                f"Hidden Layer {i+1} shape: {self.data.shape} \n" \
                                                f"Hidden Layer {i+1} Weights: {self.hidden_layer_weights[self.hidden_layers - 1]} \n" \
                                                f"Hidden Layer {i+1} Bias: {self.hidden_layer_bias[self.hidden_layers - 1]} \n" \
                                                f"Hidden Layer {i+1} Function: {self.hidden_layer_function[i].__name__} \n\n"

    def train(self, learning_rate):
        self.architecture()
        error = mean_squared_error(self.target, self.output_layer)
        base_error = error
        last_chart_error = error
        i = 0
        no_improvement = 0
        while 1 == 1:
            i += 1
            self.back_prop(learning_rate)
            new_error = mean_squared_error(self.sample_target, self.output_layer)
            percentage = new_error/last_chart_error
            percentage = (percentage-0.98)/0.02
            print(i, new_error, percentage)

            if error - new_error <= base_error * 0.0005 or error < 0.0005:
                no_improvement += 1
            else:
                no_improvement = 0
                error = new_error

            if new_error <= last_chart_error * 0.98 or i <= 5:
                nn.make_chart(i)
                last_chart_error = new_error

            if no_improvement > 100 and i > 1000:
                nn.make_chart(i)
                self.run_with_set_parameters()
                break
        self.architecture()



clear_folder("Charts/MLP/")
#x, y = simple_classifier_data(2000, 1, 10, "MLP")
np.random.seed(1)
x, y = complex_classifier_data(2000, 1, 30, "MLP")
nn = neural_network(x, y)
nn.add_hidden_layer(4, ReLU)
nn.add_output_layer()

nn.train(0.1)

