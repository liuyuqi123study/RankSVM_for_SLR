import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(20, 2)  # 20 samples, 2 features
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # Class 1 (higher rank)
              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])  # Class -1 (lower rank)
from sklearn.svm import SVC

# Train an SVM model
model = SVC(kernel='linear')
model.fit(X, y)

import matplotlib.pyplot as plt

# Create a mesh to plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict the function value for the mesh
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the contour and training examples
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap=plt.cm.coolwarm, edgecolors='k')

plt.title('Rank-SVM Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
