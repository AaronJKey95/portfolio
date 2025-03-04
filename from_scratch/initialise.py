import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def clear_folder(folder_path):
    # List all files and subdirectories in the folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        # If it's a file, remove it
        if os.path.isfile(item_path):
            os.remove(item_path)


def simple_classifier_data(n_samples, n_features, class_sep, name):
    np.random.seed(42)
    y = np.random.randint(0, 2, size=n_samples)
    x = np.empty((n_samples, n_features))
    mean_class0 = np.zeros(n_features)
    mean_class1 = np.ones(n_features) * class_sep
    cov = np.eye(n_features)
    for i in range(n_samples):
        if y[i] == 0:
            x[i, :] = np.random.multivariate_normal(mean_class0, cov)
        else:
            x[i, :] = np.random.multivariate_normal(mean_class1, cov)

    make_chart(x, y, name, first_chart=True)
    return x, y


def complex_classifier_data(n_samples, n_features, class_sep, name):
    y = np.random.randint(0, 2, size=n_samples)
    x = np.empty((n_samples, n_features))
    mean_class0 = np.zeros(n_features)
    mean_class1 = np.ones(n_features) * class_sep
    mean_class2 = np.ones(n_features) * class_sep*2
    mean_class3 = np.ones(n_features) * class_sep*3
    mean_class4 = np.ones(n_features) * class_sep*4
    mean_class5 = np.ones(n_features) * class_sep*5
    cov = np.eye(n_features)
    for i in range(n_samples):
        if y[i] == 0:
            if np.random.randint(0, 3) != 0:
                x[i, :] = np.random.multivariate_normal(mean_class0, cov)
            elif np.random.randint(0, 2) == 0:
                x[i, :] = np.random.multivariate_normal(mean_class2, cov)
            else:
                x[i, :] = np.random.multivariate_normal(mean_class2, cov)
        else:
            if np.random.randint(0, 3) != 0:
                x[i, :] = np.random.multivariate_normal(mean_class1, cov)
            elif np.random.randint(0, 2) == 0:
                x[i, :] = np.random.multivariate_normal(mean_class1, cov)
            else:
                x[i, :] = np.random.multivariate_normal(mean_class1, cov)

    make_chart(x, y, name, first_chart=True)
    return x, y


def make_chart_2d(x, y, name, func, param, error, i, first_chart):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, edgecolor='k', s=20)
    if func is not None:
        x_func = np.linspace(min(x), max(x), 100000)
        if param is None:
            y_func = func(chart_data=x_func.copy())
        else:
            y_func = func(x_func, param)
        plt.plot(x_func, y_func, color='red')
    if error is not None and param is not None:
        plt.text(0.95, 0.05, f"Iter. {i} Error: {error:.4f}, b = {param[0]:.2f}, w = {param[1]:.2f}",
                 transform=plt.gca().transAxes,  # Use Axes Coordinates
                 fontsize=12, color='white',
                 ha='right', va='bottom',
                 bbox=dict(facecolor='black', alpha=0.7, edgecolor='white'))
    elif error is not None:
        plt.text(0.95, 0.05, f"Iter. {i} Error: {error:.4f}",
                 transform=plt.gca().transAxes,  # Use Axes Coordinates
                 fontsize=12, color='white',
                 ha='right', va='bottom',
                 bbox=dict(facecolor='black', alpha=0.7, edgecolor='white'))
    #plt.ylim(-0.1, 1.1)
    plt.xlabel("x")
    plt.ylabel("y")
    if first_chart:
        plt.savefig(f"Charts/{name}/learning_plot_data.png")
    else:
        plt.savefig(f"Charts/{name}/learning_plot_i{str(i)}.png")
    plt.close()


def make_chart_3d(x, y, name, func, param, error, i, first_chart):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of data points
    ax.scatter(x[:, 0], x[:, 1], y, edgecolor='k', s=20)

    if func is not None:
        # Create a mesh grid for 3D function surface
        x1_range = np.linspace(min(x[:, 0]), max(x[:, 0]), 50)
        x2_range = np.linspace(min(x[:, 1]), max(x[:, 1]), 50)
        X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)

        # Compute function values on grid
        if param is not None:
            Z_grid = func(np.column_stack((X1_grid.ravel(), X2_grid.ravel())), param).reshape(X1_grid.shape)
        else:
            Z_grid = func(chart_data=np.column_stack((X1_grid.ravel(), X2_grid.ravel()))).reshape(X1_grid.shape)
        # Plot the surface
        ax.plot_surface(X1_grid, X2_grid, Z_grid, color='red', alpha=0.6)

    if error is not None and param is not None:
        ax.text2D(0.05, 0.95,
                  f"Iter. {i} Error: {error:.4f}, b = {param[0]:.2f}, w1 = {param[1]:.2f}, w2 = {param[2]:.2f}",
                  transform=ax.transAxes, fontsize=12, color='white',
                  bbox=dict(facecolor='black', alpha=0.7, edgecolor='white'))
    elif error is not None:
        ax.text2D(0.05, 0.95,
                  f"Iter. {i} Error: {error:.4f}",
                  transform=ax.transAxes, fontsize=12, color='white',
                  bbox=dict(facecolor='black', alpha=0.7, edgecolor='white'))
    #plt.ylim(-0.1, 1.1)
    ax.set_xlabel("Feature 1 (X1)")
    ax.set_ylabel("Feature 2 (X2)")
    ax.set_zlabel("Target (y)")
    if first_chart:
        plt.savefig(f"Charts/{name}/learning_plot_data.png")
    else:
        plt.savefig(f"Charts/{name}/learning_plot_i{str(i)}.png")
    plt.close()


def make_chart(x, y, name, func=None, param=None, error=None, i=1, first_chart=False):
    if x.shape[1] == 1:
        make_chart_2d(x, y, name=name, func=func, param=param, error=error, i=i, first_chart=first_chart)
    elif x.shape[1] == 2:
        make_chart_3d(x, y, name=name, func=func, param=param, error=error, i=i, first_chart=first_chart)
    else:
        print("Unable to visualise more than 3 dimensions")


def deep_copy_dict(d):
    if isinstance(d, dict):  # If it's a dictionary
        return {k: deep_copy_dict(v) for k, v in d.items()}
    return d  # Return the value if it's not a dictionary (base case)
