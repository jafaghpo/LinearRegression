import csv
import math


def normalize_float(value: float, min: float, max: float) -> float:
    """Normalize a float value.

    Args:
        value: A float value to be normalized.
        min: The minimum value in the list.
        max: The maximum value in the list.

    Returns:
        A normalized copy of the float value.
    """
    return (value - min) / (max - min)


def denormalize_float(value: float, min: float, max: float) -> float:
    """Denormalize a normalized float value.

    Args:
        value: A normalized float value to be denormalized.
        min: The minimum value in the original list of values.
        max: The maximum value in the original list of values.

    Returns:
        A denormalized copy of the float value.
    """
    return value * (max - min) + min


def normalize_data(values: list[float], min: float, max: float) -> list[float]:
    """Normalize a list of float values.

    Args:
        values: A list of float values to be normalized.
        min: The minimum value in the list.
        max: The maximum value in the list.

    Returns:
        A normalized copy of the list of values.
    """
    return [normalize_float(x, min, max) for x in values]


def denormalize_data(values: list[float], min: float, max: float) -> list[float]:
    """Denormalize a list of normalized float values.

    Args:
        values: A list of normalized float values to be denormalized.
        min: The minimum value in the original list of values.
        max: The maximum value in the original list of values.

    Returns:
        A denormalized copy of the list of values.
    """
    return [denormalize_float(x, min, max) for x in values]


def denormalize_theta(
    theta: list[float], y_min: int, y_max: int, x_min: int, x_max: int
) -> list[float]:
    """Denormalize the thetas.

    Args:
        theta: A list of normalized thetas to be denormalized.
        y_min: The minimum value in the original list of prices.
        y_max: The maximum value in the original list of prices.
        x_min: The minimum value in the original list of mileages.
        x_max: The maximum value in the original list of mileages.

    Returns:
        A denormalized copy of the list of thetas.
    """
    theta[0] = theta[0] * (y_max - y_min) + y_min
    theta[1] = theta[1] * (y_max - y_min) / (x_max - x_min)
    return theta


def predict(mileage: float, theta: list[float]) -> float:
    """Predict the price of a car based on its mileage using the given thetas.

    Args:
        mileage: A float representing the mileage of the car.
        theta: A list of floats representing the weights or thetas of the trained model.

    Returns:
        A float representing the predicted price of the car.
    """
    return theta[0] + theta[1] * mileage


def load_data(filename: str) -> tuple[list[float], list[float]]:
    """Load the data from the given file and return it as two lists, one for the
        mileages and another for the prices.

    Args:
        filename: The name of the file containing the data.

    Returns:
        A tuple containing two lists, one for the mileages and another for the prices.
    """
    mileages = []
    prices = []

    with open(filename, "r") as f:
        # Skip the header row
        next(f)
        for line in f:
            km, price = line.strip().split(",")
            mileages.append(float(km))
            prices.append(float(price))

    return mileages, prices


def load_theta(filename: str) -> list[float]:
    """Load the thetas from the given file and return them as a list.

    Args:
        filename: The name of the file containing the thetas.

    Returns:
        A list containing the thetas.
    """
    try:
        with open(filename, "r") as f:
            reader = csv.reader(f)
            try:
                row = next(reader)
                return [float(row[0]), float(row[1])]
            except (ValueError, TypeError):
                print("Incorrect format of theta values, using default thetas (0, 0)")
                return [0, 0]
    except FileNotFoundError:
        print("Theta file not found, using default thetas (0, 0)")
        return [0, 0]


def get_mileage():
    """
    Get the mileage from the user.
    """

    while True:
        try:
            mileage = float(input("Enter the mileage: "))
            if mileage < 0:
                raise ValueError("Mileage can't be negative")
            if mileage > 999999999:
                raise ValueError("Mileage is too large")
            return mileage
        except ValueError as e:
            print(str(e).capitalize())
        except (KeyboardInterrupt, SystemExit):
            print("\nInterrupted")
            exit(0)


def main():
    # Read the thetas from the CSV file
    theta = load_theta("theta.csv")

    # Check if the thetas are valid (not nan)
    if math.isnan(theta[0]) or math.isnan(theta[1]):
        print("Incorrect format of theta values")
        return
    mileage = get_mileage()
    price = predict(mileage, theta)
    if price < 0:
        price = 0
    print(f"Predicted price for a car with {int(mileage)}km: {int(price)}$")


if __name__ == "__main__":
    main()
