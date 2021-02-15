# Titanic-Kaggle-

Problem Statement:-
Complete the analysis of what sorts of people were likely to survive.
In particular, we ask you to apply the tools of machine learning to predict which passengers survived the Titanic tragedy.

For Dataset:- https://www.kaggle.com/c/titanic/data

Dataset:
The original data has been split into two groups : training dataset(70%) and test dataset(30%).The training set is used to build our
machine learning models. The training set includes our target variable, passenger survival status along with other independent features like
gender, class, fare, and Pclass. The test set should be used to see how well our model performs on unseen data. The test set does not provide
passengers survival status. We are going to use our model to predict passenger survival status. 
For each passenger in the test set,we use the model we trained to predict whether or not they survived the sinking of the Titanic.

Algorithms Used:-
As this is a classification problem so we have used different classification algorithms like:
1. KNearestNeighbours
2. DecisionTreeClassifier
3. RandomForestClassifier
4. SVM

Results:
After training with the algorithms , we have validate our trained algorithms with test data set and measure the algorithms performance with
godness of fit with confusion matrix for validation. 70% of data as training data set and 30% as testing data set. The accuracy of predicting
the survival rate using decision tree algorithm(79.8%) is high when compared with rest of the algorithms used for the given data set.

Conclusion:
The analysis revealed interesting patterns across individual-level features. Factors such as socioeconomic status, social norms and family
composition appeared to have an impact on likelihood of survival. 
