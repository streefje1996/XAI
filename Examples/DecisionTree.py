from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from XAI.XAI import XAI

if __name__ == "__main__":
    df = pd.DataFrame(pd.read_csv("./Data/wine/winequality-red.csv", sep=','))

    df = df.dropna()

    X = df[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]
    y = df[["quality"]]

    X_train = X[:-100]
    y_train = y[:-100]

    X_score = X[-100:]
    y_score = y[-100:]

    regr = tree.DecisionTreeRegressor()

    regr.fit(X_train,y_train)       

    xai = XAI(regr.predict,X_train,y_train)

    xai.plot_global(0,link="identity", task="regression", summarise_background=True, n_background_samples=20)

    plt.show()
    
    exit(0)
exit(1)


