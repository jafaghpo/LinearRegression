import random
import matplotlib.pyplot as plt
import numpy as np

from predict import (
    predict,
    load_data,
    normalize_data,
    denormalize_float,
    denormalize_data,
)


def train(
    mileages: list[float], prices: list[float], learning_rate: float, max_iter: int
) -> dict[str, list[tuple[float, float] | float]]:
    """
    Trains a linear regression model using the gradient descent algorithm
    on the given dataset.

    Args:
        mileages: A list of mileage values.
        prices: A list of price values.
        learning_rate: The learning rate for the gradient descent algorithm.
        max_iter: The maximum number of iterations to run
                    the gradient descent algorithm for.

    Returns:
        A dictionary containing the evolution of the theta values, learning rate,
        and error over time.
    """
    theta = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]
    record = {"theta": [], "learning_rate": [], "error": []}

    for i in range(max_iter):
        theta, learning_rate, error = gradient_descent(
            mileages, prices, theta, learning_rate
        )

        if i > 0:
            # Adjust learning rate using the bold driver algorithm
            if error > record["error"][-1]:
                learning_rate /= 2
            else:
                learning_rate *= 1.05

        record["theta"].append(tuple(theta))
        record["learning_rate"].append(learning_rate)
        record["error"].append(error)

        if i > 0 and abs(record["error"][-1] - record["error"][-2]) < 1e-5:
            break

    return record


def gradient_descent(
    mileages: list[float], prices: list[float], theta: list[float], learning_rate: float
) -> tuple[list[float], float, float]:
    """
    Performs one iteration of the gradient descent algorithm on the given dataset.

    Args:
        mileages: A list of mileage values.
        prices: A list of price values.
        theta: A list of theta values.
        learning_rate: The learning rate for the gradient descent algorithm.

    Returns:
        A tuple containing the updated theta values, the learning rate, and the error.
    """
    m = len(mileages)
    theta_gradient = [0, 0]
    error = 0
    for mileage, price in zip(mileages, prices):
        prediction = predict(mileage, theta)
        error += (price - prediction) ** 2
        theta_gradient[0] += prediction - price
        theta_gradient[1] += (prediction - price) * mileage
    error /= m
    theta_gradient[0] /= m
    theta_gradient[1] /= m
    theta[0] -= learning_rate * theta_gradient[0]
    theta[1] -= learning_rate * theta_gradient[1]
    return theta, learning_rate, error


def mean_squared_error(
    mileages: list[float], prices: list[float], theta: list[float]
) -> float:
    """
    Calculates the mean squared error of the given model on the given dataset.

    Args:
        mileages: A list of mileage values.
        prices: A list of price values.
        theta: A list of theta values.

    Returns:
        The mean squared error of the given model on the given dataset.
    """
    m = len(mileages)
    error = 0
    for mileage, price in zip(mileages, prices):
        prediction = predict(mileage, theta)
        error += (price - prediction) ** 2
    return error / m


def plot_results(
    mileages: list[float],
    prices: list[float],
    record: dict[str, list[tuple[float] | float]],
):
    """
    Plots the results of the gradient descent algorithm.

    Args:
        mileages: A list of mileage values.
        prices: A list of price values.
        record: A dictionary containing the evolution of the error, the learning rate,
                and theta values.
    """
    # Create the figure and subplots
    fig = plt.figure(figsize=(11, 11))
    ax = fig.subplots(2, 2)

    # Plot the best fit line
    min_x, max_x = min(mileages), max(mileages)
    min_y, max_y = min(prices), max(prices)
    final_theta = record["theta"][-1]
    ax[0][0].plot(
        [min_x, max_x], [predict(min_x, final_theta), predict(max_x, final_theta)], "r"
    )
    ax[0][0].scatter(mileages, prices)
    ax[0][0].set_title("Best fit line")
    ax[0][0].set_xlim(min_x - 0.1, max_x + 0.1)
    ax[0][0].set_ylim(min_y - 0.1, max_y + 0.1)

    # Plot the error over time
    ax[0][1].plot(range(len(record["error"])), record["error"])
    ax[0][1].set_title("Error over time")
    ax[0][1].set_xlabel("Iteration")
    ax[0][1].set_ylabel("Error")

    # Plot the learning rate over time
    ax[1][0].plot(range(len(record["learning_rate"])), record["learning_rate"])
    ax[1][0].set_title("Learning rate over time")
    ax[1][0].set_xlabel("Iteration")
    ax[1][0].set_ylabel("Learning rate")

    # Plot the theta values over time
    ax[1][1].plot(
        range(len(record["theta"])), [t[0] for t in record["theta"]], label="Theta 0"
    )
    ax[1][1].plot(
        range(len(record["theta"])), [t[1] for t in record["theta"]], label="Theta 1"
    )
    ax[1][1].set_title("Theta values over time")
    ax[1][1].set_xlabel("Iteration")
    ax[1][1].set_ylabel("Theta value")
    ax[1][1].legend()

    # Plotting the error surface in 3D

    # Create arrays of 100 evenly spaced values between -10 and 10
    x = np.linspace(-10, 10, num=100)
    y = np.linspace(-10, 10, num=100)

    # Create a meshgrid of the x and y values
    # The grid is created so that the values of X and Y can be used to evaluate
    # the mean squared error at each point in the grid
    X, Y = np.meshgrid(x, y)

    Z = np.array(
        [
            mean_squared_error(np.array([x_val, y_val]), prices, mileages)
            for x_val, y_val in zip(np.ravel(X), np.ravel(Y))
        ]
    )
    Z = Z.reshape(X.shape)

    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel("Theta 0")
    ax.set_ylabel("Theta 1")
    ax.set_zlabel("Mean Squared Error")
    plt.title("Error Surface")

    # Show the plot
    plt.show()


def main():
    mileages, prices = load_data("data.csv")
    normalized_mileages = normalize_data(mileages, min(mileages), max(mileages))
    normalized_prices = normalize_data(prices, min(prices), max(prices))
    learning_rate = 0.01
    max_iter = 1000
    record = train(normalized_mileages, normalized_prices, learning_rate, max_iter)
    theta = [
        denormalize_data(theta, min(prices), max(prices)) for theta in record["theta"]
    ]
    record["theta"] = theta
    with open("theta.csv", "w") as f:
        f.write(",".join(map(str, record["theta"][-1])))

    plot_results(normalized_mileages, normalized_prices, record)


if __name__ == "__main__":
    main()
