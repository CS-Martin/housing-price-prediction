import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define the hypothesis function
def hyp(theta, x):
    return theta[0] + theta[1] * x

# Define the cost function
def cost(theta, data):
    return np.sum((hyp(theta, data[:, 0]) - data[:, 1]) ** 2) / (2 * len(data))

# Define the derivatives with respect to theta_0 and theta_1
def der0(theta, data):
    return np.sum(hyp(theta, data[:, 0]) - data[:, 1]) / len(data)

def der1(theta, data):
    return np.sum((hyp(theta, data[:, 0]) - data[:, 1]) * data[:, 0]) / len(data)

# Perform gradient descent to learn theta
def gradient_descent(data, theta, alpha, num_iters):
    m = len(data)
    cost_history = []
    for i in range(num_iters):
        theta[0] -= alpha * der0(theta, data)
        theta[1] -= alpha * der1(theta, data)
        cost_history.append(cost(theta, data))
    return theta, cost_history

# Load and preprocess the data
def load_data():
    data = np.loadtxt("data/realestate.csv", dtype="float", delimiter=",")
    return data[:, [4, 7]]  # Selecting the 5th and 8th columns as features and target

# Plotting function
def plot_regression_line(data, theta):
    x = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
    y = hyp(theta, x)
    plt.scatter(data[:, 0], data[:, 1], label="Data Points")
    plt.plot(x, y, color="red", label="Regression Line")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    st.pyplot(plt)

# Streamlit app
def main():
    st.title('Linear Regression with Gradient Descent')
    
    # Sidebar for hyperparameters
    alpha = st.sidebar.slider("Learning Rate", min_value=0.00001, max_value=0.01, value=0.002, step=0.00001)
    iterations = st.sidebar.slider("Iterations", min_value=1000, max_value=50000, value=10000, step=1000)
    
    # Button to run gradient descent
    if st.button('Run Gradient Descent'):
        # Load data
        data = load_data()
        
        # Initialize theta
        theta = [0, 0]
        
        # Run gradient descent
        theta, cost_history = gradient_descent(data, theta, alpha, iterations)
        
        # Show the results
        st.write(f"Optimized Theta: {theta}")
        st.line_chart(cost_history)
        
        # Plot the data and the regression line
        plot_regression_line(data, theta)

if __name__ == "__main__":
    main()
