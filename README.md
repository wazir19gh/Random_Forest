# Random_Forest Regressor

This .py file would describe the internals of sampling with replacement i.e. bootstrapping, column sampling and fitting of n base learners which are fully grown decision trees, calculation of training mse and oob score. OOP concept would be used to define a class Random Forest wherein there would be various methods which would be doing the various tasks of Random Forests as in preparation of inputs with row and column sampling, training and calculation of training mse and OOB score or validation score.

The .py file is named as RF_rgsr.py which consists of the following:-

1) Class Random_Forest_Regressor which is initialized with the input data x, target data y , the number of base learners and a boolean flag indicating whether column sampling would be used or not.
2) Method bootstrapping which would create a bootstrap sample by sampling with replacement along with column sampling with 3 to total number of columns if the user requires.
3) Method train which would train n decision trees, would determine the training mse and would print it.
4) Method oob_score whoch would get the oob samples and would get the oob score or mse on those samples.
5) Method rf_predict which would take one query point as input and would return the prediction on the trained Random Forest Regressor.

Please import this rf_rgsr.py file in any program and create an instance of the class Random_Forest_Regressor and call the methods defined in the classes to train the Random_Forest_Regressor, get the oob_score and also the prediction on a test data point.

# Random_Forest Classifier

This .py file would describe the internals of sampling with replacement i.e. bootstrapping, column sampling and fitting of n base learners which are fully grown decision trees, calculation of training mse and oob score. OOP concept would be used to define a class Random Forest wherein there would be various methods which would be doing the various tasks of Random Forests as in preparation of inputs with row and column sampling, training and calculation of training mse and OOB score or validation score.

The .py file is named as RF_classifier.py which consists of the following:-

1) Class Random_Forest_Classifier which is initialized with the input data x, target data y , the number of base learners and a boolean flag indicating whether column sampling would be used or not.
2) Method bootstrapping which would create a bootstrap sample by sampling with replacement along with column sampling with 3 to total number of columns if the user requires.
3) Method train which would train n decision trees, would determine the training accuracy and would print it.
4) Method oob_score which would get the oob samples and would get the oob score or accuracy on those samples.
5) Method rf_predict which would take one query point as input and would return the prediction on the trained Random Forest Classifier.

Please import this rf_classifier.py file in any program and create an instance of the class Random_Forest_Classifier and call the methods defined in the classes to train the Random_Forest_Classifier, get the oob_score and also the prediction on a test data point.


