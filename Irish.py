import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the iris dataset
@st.cache_data
def load_data():
    iris = load_iris()
    return iris.data, iris.target, iris.target_names

X, y, target_names = load_data()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Streamlit app
st.title('Multi-Layer Perceptron Classifier for Iris Dataset')

# Sidebar for hyperparameters
st.sidebar.header('Hyperparameters')
hidden_layer_sizes = st.sidebar.text_input('Hidden Layer Sizes (comma-separated)', '10,5')
max_iter = st.sidebar.slider('Max Iterations', 100, 2000, 1000)
activation = st.sidebar.selectbox('Activation Function', ['relu', 'tanh', 'logistic'])

# Convert hidden_layer_sizes to tuple
hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(',')))

# Create and train the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, activation=activation, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Make predictions
y_pred = mlp.predict(X_test_scaled)

# Display results
st.header('Results')
st.subheader('Classification Report')
st.text(classification_report(y_test, y_pred, target_names=target_names))

st.subheader('Confusion Matrix')
st.text(confusion_matrix(y_test, y_pred))

# Plot the decision boundaries
def plot_decision_boundaries(X, y, model, scaler):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel(), 
                                             np.zeros_like(xx.ravel()), 
                                             np.zeros_like(xx.ravel())]))
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_title('MLP Decision Boundaries')
    return fig

st.subheader('Decision Boundaries')
fig = plot_decision_boundaries(X[:, :2], y, mlp, scaler)
st.pyplot(fig)

# Display model accuracy
st.sidebar.subheader('Model Accuracy')
accuracy = mlp.score(X_test_scaled, y_test)
st.sidebar.write(f'Accuracy: {accuracy:.2f}')

# Visualize MLP architecture
def plot_mlp_architecture(mlp):
    fig, ax = plt.subplots(figsize=(12, 8))
    n_layers = len(mlp.coefs_) + 1
    layer_sizes = [X.shape[1]] + list(mlp.hidden_layer_sizes) + [3]
    
    for i in range(n_layers):
        layer_size = layer_sizes[i]
        x = np.ones(layer_size) * i
        y = np.linspace(0, 1, layer_size)
        ax.scatter(x, y, s=100, c='blue', zorder=2)
        
        if i < n_layers - 1:
            for j in range(layer_size):
                for k in range(layer_sizes[i+1]):
                    ax.plot([i, i+1], [y[j], np.linspace(0, 1, layer_sizes[i+1])[k]], 
                            c='gray', alpha=0.2, zorder=1)
    
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(['Input'] + [f'Hidden {i+1}' for i in range(n_layers-2)] + ['Output'])
    ax.set_ylabel('Neurons')
    ax.set_title('MLP Architecture')
    plt.tight_layout()
    return fig

st.subheader('MLP Architecture')
arch_fig = plot_mlp_architecture(mlp)
st.pyplot(arch_fig)