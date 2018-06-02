# Titanic
Predict if a passenger survived the sinking of the Titanic or not

Follow CRISP-DM process

1) Problem understanding
2) Data understanding
3) Exploratory Data analysis
4) Data modelling
5) Model evaluation


PROBLEM UNDERSTANDING____________________
The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
In this challenge, to complete the analysis of what sorts of people were likely to survive. In particular, applying the tools of machine learning to predict which passengers survived the tragedy.

Goal - Predict if a passenger survived the sinking of the Titanic or not. 
For each PassengerId in the test set, you must predict a 0 or 1 value for the Survived variable.


DATA UNDERSTANDING___________________
The data has been split into two groups:
•	training set (train.csv)
•	test set (test.csv)


The training set should be used to build your machine learning models. For the training set, provided the outcome (also known as the “ground truth”) for each passenger. The model will be based on “features” like passengers’ gender and class.
The test set should be used to see how well model performs on unseen data. For the test set, it is not provided the ground truth for each passenger. It is our job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.
Also included gender_submission.csv, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.


Data Dictionary
Variable	Definition	Key
survival	Survival	0 = No, 1 = Yes
pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	Sex	
Age	Age in years	
sibsp	# of siblings / spouses aboard the Titanic	
parch	# of parents / children aboard the Titanic	
ticket	Ticket number	
fare	Passenger fare	
cabin	Cabin number	
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
Variable Notes
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5





sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.



Exploratory Data analysis -------------
 Data modelling----------------------
 Model evaluation--------------------         {THESE 3 ARE DONE IN SOURCE CODE *MENTIONED IN COMMeNTS}
