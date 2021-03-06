Linear regression example
=========================
Creating a simple explainer for linear regression

.. code-block:: python

	from sklearn import linear_model
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from XAI.XAI import XAI

	#load dataset: https://www.kaggle.com/uditkhanna112/startups-in-usamultiple-linear-regression
	df = pd.DataFrame(pd.read_csv("./50_Startups.csv", sep=','))   
	
	#prepare dataset
	df = df.dropna()
	
	X = df[["R&D Spend","Administration","Marketing Spend","State"]]
	y = df[["Profit"]] 
	X = pd.get_dummies(X)
	
	#split dataset in training and testing (score)
	X_train = X[:-10]
	y_train = y[:-10]
	
	X_score = X[-10:]
	y_score = y[-10:]
	
	#create a linear regression model
	regr = linear_model.LinearRegression()
	
	#fit the model
	regr.fit(X_train,y_train)
	
	#create the XAI object using: the predict function (regr.predict), the training input set (X_train), and training output set (y_train)
	xai = XAI(regr.predict, X_train, y_train)

From here you can:

* Plot a global explanation
* Plot a local explanation
* Plot a curve (global)

All of the above explanations write to the matplotlib.pyplot.

Plot global
-----------

.. code-block:: python
	
	#Plot a global explanation, using the link identity and task regression for regression models.
	#More info about Link can be foun here: https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function
	xai.plot_global(link="identity", task="regression")
	
	#use plt.show() to display result
	plt.show()
	
Plot local
----------

.. code-block:: python
	
	#Plot a local explanation, using the link identity and task regression for regression models.
	#More info about Link can be foun here: https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function
	#The first argument is the instance to be explained (X_score.iloc[0])
	#The plot_type argument can be set to "waterfall" or "force". Note if unset default is "force".
	xai.plot_local(X_score.iloc[0],link="identity", task="regression", plot_type="waterfall")
	
	#use plt.show() to display result
	plt.show()
	
Plot curve
----------

.. code-block:: python
	
	#Plot a global explanation curve.
	xai.plot_curve()
	
	#use plt.show() to display result
	plt.show()

	
	
