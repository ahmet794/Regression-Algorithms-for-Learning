#plotting.py, handles all the plotting functionalities.
import linear_regression as lr
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasAgg as FigureCanvas


def plot_linear_r(data):
    """
    Plot the linear regression using a simple data.

    :param data: two dimensional array with two columns

    :return: The plot of the graph with data points and the best fit line.
    """
    x = data[:, 0]
    y = data[:, 1]

    b = lr.slope(data)
    a = lr.intercept(data)

    x_line = np.linspace(min(x), max(x), 100) # Create a space so that the line can be formed.
    y_line = b * x_line + a

    plt.scatter(x, y, color='blue', label='Data Points')# Plot data points.
    plt.plot(x_line, y_line, color='red', label=f'y = {b}x + {a}')# Plot the line.
    plt.xlabel("Temperature")
    plt.ylabel("Percentage of Fire")
    plt.legend()
    plt.title('Simple Linear Regression Plot')
    plt.grid(True)
    plt.show()



def plt_ridge_lr_comparison(lr_coefs, r_coefs, alpha):
    """
    Plot a comparison graph of LR and RR coefficients.

    :param lr_coefs: A list of LR model coefficients.
    :param r_coefs: A list of RR model coefficients.
    :param alpha: A float for the tuning parameter.

    :return: Shows the bar graph.
    """
    indices = np.arange(len(lr_coefs[1:]))

    plt.figure(figsize=(12, 6))
    width = 0.35

    plt.bar(indices - width/2, lr_coefs[1:], width=width, label="Linear Regression Ceofficients", color = 'blue')
    plt.bar(indices + width/2, r_coefs[1:], width=width, label='Ridge Regression Coefficients', color = "green")

    plt.xlabel("Coefficient Index")
    plt.ylabel("Coefficient Value")
    plt.title(f"Comparison of Linear Regression and Ridge Regression Coefficients, $\lambda={alpha}$")
    plt.legend()

    plt.show()

def model_coefficients(coefs, title='Model Coefficients'):
    """
    Creates a plot of the model's coefficients, saves it as a PNG image directly from the Figure object,
    and returns a QLabel containing the image.
    """
    # Create the figure and axis
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.bar(range(len(coefs)), coefs)
    tick_indices = range(0, len(coefs), max(1, len(coefs) // 5))
    ax.set_xticks(list(tick_indices))
    ax.set_xticklabels([str(i) for i in tick_indices], rotation=45, ha='right')
    ax.set_xlabel('Coefficient Index')
    ax.set_ylabel('Coefficient Value')
    ax.set_title(title)

    # Adjust the layout to fit the window
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.9)  # Adjust these as needed
    fig.tight_layout()  

    # Use a canvas to draw the figure's content
    canvas = FigureCanvas(fig)
    canvas.draw()

    # Save the figure to a BytesIO buffer
    buf = BytesIO()
    canvas.print_png(buf)

    # Seek to the start of the buffer
    buf.seek(0)

    # Load the image from the buffer into a QPixmap
    pixmap = QPixmap()
    pixmap.loadFromData(buf.read(), 'PNG')

    # Create and return a QLabel containing the QPixmap
    label = QLabel()
    label.setPixmap(pixmap)

    return label