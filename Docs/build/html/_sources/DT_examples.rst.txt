Decision-tree regression example
================================
Creating a simple explainer for decision-tree regression

.. code-block:: python

	from sklearn import tree
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt	
	from XAI.XAI import XAI
	
	#load dataset: https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009
	df = pd.DataFrame(pd.read_csv("./winequality-red.csv", sep=','))   
	
	#prepare dataset
	df = df.dropna()
	
	X = df[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
	y = df[["quality"]]
	
	#split dataset in training and testing (score)
	X_train = X[:-100]
	y_train = y[:-100]
	
	X_score = X[-100:]
	y_score = y[-100:]
	
	#create a decision-tree regression model
	regr = tree.DecisionTreeRegressor()
	
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
	xai.plot_local(X_score.iloc[0], link="identity", task="regression", plot_type="waterfall")
	
	#use plt.show() to display result
	plt.show()
	
Plot curve
----------

.. code-block:: python
	
	#Plot a global explanation curve.
	xai.plot_curve()
	
	#use plt.show() to display result
	plt.show()

	
	
