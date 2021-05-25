from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from XAI.XAI import XAI

if __name__ == "__main__":
    df = pd.DataFrame(pd.read_csv("./Data/startup/50_Startups.csv", sep=','))

    df = df.dropna()

    X = df[["R&D Spend", "Administration", "Marketing Spend", "State"]]
    y = df[["Profit"]] 
    X = pd.get_dummies(X)

    X_train = X[:-10]
    y_train = y[:-10]

    X_score = X[-10:]
    y_score = y[-10:]

    regr = linear_model.LinearRegression()

    regr.fit(X_train,y_train)

    xai = XAI(regr.predict, X_train, y_train)
    xai.plot_curve()

    plt.show()

    exit(0)
exit(1)

