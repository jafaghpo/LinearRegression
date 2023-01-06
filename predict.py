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

    with open(filename, 'r') as f:
        # Skip the header row
        next(f)
        for line in f:
            km, price = line.strip().split(',')
            mileages.append(float(km))
            prices.append(float(price))
    
    return mileages, prices

def main():
    # Read the thetas from the CSV file
    try:
        with open('theta.csv') as f:
            reader = csv.reader(f)
            try:
                # Read the thetas from the first (and only) row of the file
                row = next(reader)
                theta = [float(row[0]), float(row[1])]
            except (ValueError, TypeError):
                # The file is badly formatted
                print('Incorrect format of theta values')
                return
    except FileNotFoundError:
        # The file does not exist
        print('No training data available')
        return
    
    # Check if the thetas are valid (not nan)
    if math.isnan(theta[0]) or math.isnan(theta[1]):
        print('Incorrect format of theta values')
        return
    
    # Prompt the user to enter a mileage
    try:
        mileage = float(input('Enter a mileage: '))
    except (ValueError, TypeError, SyntaxError, NameError):
        print('Incorrect format of mileage')
        return
    except EOFError:
        print('\nNo mileage entered')
        return
    except (KeyboardInterrupt, SystemExit):
        print('\nInterrupted')
        return
    if mileage < 0:
        print('A mileage cannot be negative')
        return
    mileages, prices = load_data('data.csv')
    try:
        normalized_mileage = normalize_float(mileage, min(mileages), max(mileages))
    except (ValueError, TypeError, ZeroDivisionError):
        print('The data is invalid')
        return
    # Predict the price using the thetas
    normalized_price = predict(normalized_mileage, theta)
    # Denormalize the price
    price = denormalize_float(normalized_price, min(prices), max(prices))
    if price < 0:
        price = 0
    print(f'Predicted price for a car with {int(mileage)} mileage: {int(price)}')

if __name__ == '__main__':
    main()