import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from math import isclose

# Function to read CSV data and convert to training arrays
def read_csv(csv_path):
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    X_train = data[:, :2]  # First two columns: X and Y
    Fx_train = data[:, 2]  # Third column: Fx
    Fy_train = data[:, 3]  # Fourth column: Fy
    labels = data[:, 4]  # Fifth column: Classification
    return X_train, Fx_train, Fy_train, labels

# Function to train linear regression models for Fx and Fy
def train_models(X_train, Fx_train, Fy_train):
    model_Fx = LinearRegression().fit(X_train, Fx_train)
    model_Fy = LinearRegression().fit(X_train, Fy_train)
    return model_Fx, model_Fy

# Function to predict Fx, Fy for a given input point
def predict(model_Fx, model_Fy, user_input):
    Fx_pred = model_Fx.predict([user_input])[-1]
    Fy_pred = model_Fy.predict([user_input])[-1]
    return Fx_pred, Fy_pred

# Function to calculate distance between two points
def distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# Function to detect the shape based on the coordinates
def detect_shape(coords):
    num_points = len(coords)
    
    if num_points == 2:
        return "Straight Line"

    elif num_points == 3:
        # Check if the triangle is equilateral
        d1 = distance(coords[0], coords[1])
        d2 = distance(coords[1], coords[2])
        d3 = distance(coords[2], coords[0])
        if isclose(d1, d2) and isclose(d2, d3):
            return "Equilateral Triangle"
        else:
            return "Triangle"

    elif num_points == 4:
        # Check for square or rectangle
        d1 = distance(coords[0], coords[1])
        d2 = distance(coords[1], coords[2])
        d3 = distance(coords[2], coords[3])
        d4 = distance(coords[3], coords[0])
        diag1 = distance(coords[0], coords[2])
        diag2 = distance(coords[1], coords[3])
        if isclose(d1, d2) and isclose(d2, d3) and isclose(d3, d4):
            if isclose(diag1, diag2):
                return "Square"
            else:
                return "Rhombus"
        elif isclose(d1, d3) and isclose(d2, d4):
            return "Rectangle"
        else:
            return "Quadrilateral"

    elif num_points > 4:
        # Check for circle by comparing distances from the centroid
        centroid = np.mean(coords, axis=0)
        distances = [distance(point, centroid) for point in coords]
        if isclose(max(distances), min(distances), rel_tol=0.05):
            return "Circle"
        else:
            return "Polygon"

    return "Unknown Shape"

# Function to plot the results
def plot_shapes(coords, Fx_pred, Fy_pred, shape):
    fig, ax = plt.subplots()

    # Plot the original user input points
    coords = np.array(coords)
    ax.plot(coords[:, 0], coords[:, 1], 'bo-', label='User Input Points')

    # Plot the predicted force vector as an arrow
    ax.arrow(coords[-1][0], coords[-1][1], Fx_pred - coords[-1][0], Fy_pred - coords[-1][1],
             head_width=0.2, head_length=0.3, fc='green', ec='green', label='Predicted Force Vector')

    # Display the detected shape
    ax.set_aspect('equal')
    ax.legend()
    plt.xlim([min(coords[:, 0]) - 2, max(coords[:, 0]) + 2])
    plt.ylim([min(coords[:, 1]) - 2, max(coords[:, 1]) + 2])
    plt.title(f'Detected Shape: {shape}')
    plt.show()

# Example usage
csv_path = "C:/Users/satvi/Desktop/curve/2VarRegClass.csv"  # Replace with your CSV file path

# Step 1: Read data from CSV file
X_train, Fx_train, Fy_train, labels = read_csv(csv_path)

# Step 2: Train models
model_Fx, model_Fy = train_models(X_train, Fx_train, Fy_train)

# Step 3: Take user input as coordinates
print("Enter the coordinates as X Y pairs (space-separated), one per line. Enter 'done' to finish:")
user_input = []
while True:
    coord = input()
    if coord.lower() == 'done':
        break
    user_input.append(list(map(float, coord.split())))

# Step 4: Predict Fx, Fy for the last input point
Fx_pred, Fy_pred = predict(model_Fx, model_Fy, user_input[-1])

# Step 5: Detect shape based on input
shape = detect_shape(user_input)

# Step 6: Plot the results
plot_shapes(user_input, Fx_pred, Fy_pred, shape)
