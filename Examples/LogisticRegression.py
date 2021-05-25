from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from XAI.XAI import XAI

if __name__ == "__main__":
    df = pd.DataFrame(pd.read_csv("./Data/diabetes/diabetes_data.csv", sep=','))

    df = df.dropna()

    X = df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
    y = df[["Outcome"]]

    X_train = X[:-60]
    y_train = y[:-60]

    X_score = X[-60:]
    y_score = y[-60:]

    regr = linear_model.LogisticRegression()

    regr.fit(X_train,y_train)

    xai = XAI(regr.predict_proba,X_train,y_train)

    prediction = np.array(X_score.iloc[0])
    prediction = np.reshape(prediction,(1,-1))
    prediction = regr.predict(prediction)
    print( "predicted class: ", prediction)

    xai.plot_local(X_score.iloc[0],0,plot_type="waterfall")

    plt.show()
    
    exit(0)
exit(1)

