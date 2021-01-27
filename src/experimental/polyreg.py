
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# data = pd.read_csv('data.csv')  # load data set
# data = np.random.randint(10, size=(50, 2))
# data = DataFrame(data)
# X = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
# Y = data.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column

x = np.linspace(-np.pi, np.pi, 201)
y = np.sin(x)
# plt.plot(x, y)

X = x.reshape(-1, 1)
Y = y.reshape(-1, 1)

degree = 3
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
polyreg.fit(X, Y)
Y_pred = polyreg.predict(X)

plt.figure()
plt.scatter(X,Y)
plt.plot(X, Y_pred, color="black")
plt.title("Polynomial regression with degree "+str(degree))
plt.show()

print("KOHEU.")

