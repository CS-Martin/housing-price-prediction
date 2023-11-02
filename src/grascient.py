import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def hyp(theta, x):
    return theta[0] + theta[1] * x

def cost(theta, data):
    return np.sum((hyp(theta, data[:, 0]) - data[:, 1]) ** 2) / (2 * len(data))

def der0(theta, data):
    return np.sum(hyp(theta, data[:, 0]) - data[:, 1]) / len(data)

def der1(theta, data):
    return np.sum((hyp(theta, data[:, 0]) - data[:, 1]) * data[:, 0]) / len(data)

def gradient_descent(data, theta, alpha, num_iters):
    m = len(data)
    cost_history = []
    for i in range(num_iters):
        theta[0] -= alpha * der0(theta, data)
        theta[1] -= alpha * der1(theta, data)
        cost_history.append(cost(theta, data))
    return theta, cost_history

def preprocess(df):
    # Make the transaction date as years only by removing the digits after the decimal point
    df['TransactionDate'] = df['TransactionDate'].astype(str).str.split('.').str[0].astype(int)
    return df

def load_data():
    data_path = os.path.join(os.getcwd(), 'data', 'realestate.csv') 
    columns = ['TransactionDate', 'HouseAge', 'DistanceToNearestMRTStation', 'NumberConvenienceStores', 'Latitude', 'Longitude', 'PriceOfUnitArea']
    df = pd.read_csv(data_path, names=columns)
    return preprocess(df)

def plot_regression_line(data, theta):
    x = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
    y = hyp(theta, x)
    plt.scatter(data[:, 0], data[:, 1], label="Data Points")
    plt.plot(x, y, color="red", label="Regression Line")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    st.pyplot(plt)

def main():
    df = load_data()
    st.title('Linear Regression with Gradient Descent')
    
    alpha = st.sidebar.slider("Learning Rate", min_value=0.00001, max_value=0.01, value=0.002, step=0.00001)
    iterations = st.sidebar.slider("Iterations", min_value=1000, max_value=50000, value=10000, step=1000)
    
    # Let the user select the feature and target
    feature = st.sidebar.selectbox('Feature', df.columns, index=1)
    target = st.sidebar.selectbox('Target', df.columns, index=6)

    if st.button('Run Gradient Descent'):
        # Use the selected feature and target columns
        feature_data = df[[feature]].values.reshape(-1, 1)
        target_data = df[[target]].values.reshape(-1, 1)
        data = np.hstack((feature_data, target_data))
        
        theta = [0, 0]
        
        theta, cost_history = gradient_descent(data, theta, alpha, iterations)
        
        st.write(f"Optimized Theta: {theta}")
        st.line_chart(cost_history)
        
        plot_regression_line(data, theta)

if __name__ == "__main__":
    main()
