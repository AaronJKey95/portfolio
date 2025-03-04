import numpy as np
from initialise import simple_classifier_data, make_chart, clear_folder


# initial function
def sigmoid(x, param):
    b = param[0]
    w_1 = param[1]
    if x.shape[1] == 2:
        w_2 = param[2]
        z = w_1*x[:, 0]+w_2*x[:, 1]+b
    else:
        z = w_1*x + b
    f_x = 1 / (1 + np.exp(-z))
    return f_x.flatten()


def linear(x, param):
    b = param[0]
    w_1 = param[1]
    if x.shape[1] == 2:
        w_2 = param[2]
        z = w_1*x[:, 0]+w_2*x[:, 1]+b
    else:
        z = w_1*x + b
    return z.flatten()


# loss function
def mean_squared_error(y, y_pred):
    sq_res = np.square(y_pred-y)
    mse = np.mean(sq_res)
    return mse


def gradient_descent(x, y, func, learning_rate=0.01):
    param_no = x.shape[1]+1
    param = np.zeros(param_no)
    error = mean_squared_error(y, func(x, param))
    base_error = error
    last_chart_error = error
    make_chart(x, y, "Log_Reg", func, param, error, 0)
    i = 0
    no_improvement = 0
    while 1 == 1:
        i += 1
        print(i)
        random_indices = np.random.choice(len(x), size=round(len(x)*0.1), replace=False)
        x_sample = x[random_indices]
        y_sample = y[random_indices]
        for n in range(param_no):
            d = 1e-9
            param_d = param.copy()
            param_d[n] = param_d[n]+d
            loss_1 = mean_squared_error(y_sample, func(x_sample, param))
            loss_2 = mean_squared_error(y_sample, func(x_sample, param_d))
            d_loss_tan = (loss_2-loss_1)/d
            param[n] = param[n]-d_loss_tan*learning_rate

        new_error = mean_squared_error(y, func(x, param))

        if (error - new_error <= base_error*0.000001 or error < 0.00005) and i > 1000:
            no_improvement += 1
        else:
            no_improvement = 0
            error = new_error

        if new_error <= last_chart_error * 0.98:
            make_chart(x, y, name="Log_Reg", func=func, param=param, error=new_error, i=i)
            last_chart_error = new_error

        if no_improvement > 20:
            make_chart(x, y, name="Log_Reg", func=func, param=param, error=new_error, i=i)
            break


clear_folder("Charts/Log_Reg/")
x, y = simple_classifier_data(205, 2, 10, "Log_Reg")
gradient_descent(x, y, sigmoid)
