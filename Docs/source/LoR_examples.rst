Logistic regression example
===========================
Creating a simple explainer for logistic regression

.. code-block:: python

	from sklearn import linear_model
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from XAI.XAI import XAI
	
	#load dataset: https://www.kaggle.com/kandij/diabetes-dataset
	df = pd.DataFrame(pd.read_csv("./diabetes_data.csv", sep=','))   
	
	#prepare dataset
	df = df.dropna()
	
	X = df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
	y = df[["Outcome"]]
	
	#split dataset in training and testing (score)
	X_train = X[:-60]
	y_train = y[:-60]
	
	X_score = X[-60:]
	y_score = y[-60:]
	
	#create a logistic regression model
	regr = linear_model.LogisticRegression()
	
	#fit the model
	regr.fit(X_train,y_train)
	
	#create the XAI object using: the predict function (regr.predict_proba), the training input set (X_train), and training output set (y_train)
	#the reason for using proba, is that we need to have an output of shape: (smaples * n_targets)
	xai = XAI(regr.predict_proba, X_train, y_train)

From here you can:

* Plot a global explanation
* Plot a local explanation
* Plot a curve (global)

All of the above explanations write to the matplotlib.pyplot.

Plot global
-----------

.. code-block:: python
	
	#Plot a global explanation, using the default link (logit) and task (classification).
	#More info about Link can be foun here: https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function
	xai.plot_global()
	
	#use plt.show() to display result
	plt.show()
	
Plot local
----------

.. code-block:: python

	#First get the predicted class, reshape is needed because we use 1 sample.
	prediction = regr.predict(np.reshape(X_score.iloc[0], (1, -1))
	
	#Plot a local explanation, using the default link (logit) and task (classification).
	#More info about Link can be foun here: https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function
	#The plot_type argument can be set to "waterfall" or "force". Note if unset default is "force".
	#Using the input_data (X_score.iloc[0]) and the prediction (prediction[0]), it dispays why it got to that class
	#If needed to see for other classes replace prediction[0] with class ID
	xai.plot_local(X_score.iloc[0], prediction[0], plot_type="waterfall")
	
	#use plt.show() to display result
	plt.show()
	
Plot curve
----------

.. code-block:: python
	
	#Plot a global explanation curve.
	#We provide the target_names parameter, to display on the plot.
	xai.plot_curve(target_names=["Non-diabetic", "Diabetic"])
	
	#use plt.show() to display result
	plt.show()

	
	
