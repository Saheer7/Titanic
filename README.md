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



Exploratory Data analysis_____________

We're going to consider the features in the dataset and how complete they are.

#get a list of the features within the dataset

print(train.columns)
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')

#see a sample of the dataset to get an idea of the variables
train.sample(5)

PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
397	398	0	2	McKane, Mr. Peter David	male	46.0	0	0	28403	26.0000	NaN	S
76	77	0	3	Staneff, Mr. Ivan	male	NaN	0	0	349208	7.8958	NaN	S
864	865	0	2	Gill, Mr. John William	male	24.0	0	0	233866	13.0000	NaN	S
610	611	0	3	Andersson, Mrs. Anders Johan (Alfrida Konstant...	female	39.0	1	5	347082	31.2750	NaN	S
292	293	0	2	Levy, Mr. Rene Jacques	male	36.0	0	0	SC/Paris 2163	12.8750	D	C

Numerical Features: Age (Continuous), Fare (Continuous), SibSp (Discrete), Parch (Discrete)
Categorical Features: Survived, Sex, Embarked, Pclass
Alphanumeric Features: Ticket, Cabin
What are the data types for each feature?
Survived: int
Pclass: int
Name: string
Sex: string
Age: float
SibSp: int
Parch: int
Ticket: string
Fare: float
Cabin: string
Embarked: string
Now that we have an idea of what kinds of features we're working with, we can see how much information we have about each of them.

#see a summary of the training dataset
train.describe(include = "all")
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
count	891.000000	891.000000	891.000000	891	891	714.000000	891.000000	891.000000	891	891.000000	204	889
unique	NaN	NaN	NaN	891	2	NaN	NaN	NaN	681	NaN	147	3
top	NaN	NaN	NaN	Adahl, Mr. Mauritz Nils Martin	male	NaN	NaN	NaN	CA. 2343	NaN	C23 C25 C27	S
freq	NaN	NaN	NaN	1	577	NaN	NaN	NaN	7	NaN	4	644
mean	446.000000	0.383838	2.308642	NaN	NaN	29.699118	0.523008	0.381594	NaN	32.204208	NaN	NaN
std	257.353842	0.486592	0.836071	NaN	NaN	14.526497	1.102743	0.806057	NaN	49.693429	NaN	NaN
min	1.000000	0.000000	1.000000	NaN	NaN	0.420000	0.000000	0.000000	NaN	0.000000	NaN	NaN
25%	223.500000	0.000000	2.000000	NaN	NaN	20.125000	0.000000	0.000000	NaN	7.910400	NaN	NaN
50%	446.000000	0.000000	3.000000	NaN	NaN	28.000000	0.000000	0.000000	NaN	14.454200	NaN	NaN
75%	668.500000	1.000000	3.000000	NaN	NaN	38.000000	1.000000	0.000000	NaN	31.000000	NaN	NaN
max	891.000000	1.000000	3.000000	NaN	NaN	80.000000	8.000000	6.000000	NaN	512.329200	NaN	NaN
Some Observations:
There are a total of 891 passengers in our training set.
The Age feature is missing approximately 19.8% of its values. I'm guessing that the Age feature is pretty important to survival, so we should probably attempt to fill these gaps.
The Cabin feature is missing approximately 77.1% of its values. Since so much of the feature is missing, it would be hard to fill in the missing values. We'll probably drop these values from our dataset.
The Embarked feature is missing 0.22% of its values, which should be relatively harmless.
#check for any other unusable values
print(pd.isnull(train).sum())
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
We can see that except for the abovementioned missing values, no NaN values exist.

Some Predictions:
Sex: Females are more likely to survive.
SibSp/Parch: People traveling alone are more likely to survive.
Age: Young children are more likely to survive.
Pclass: People of higher socioeconomic class are more likely to survive.



Data modelling________________

Choosing the Best Model
Splitting the Training Data
We will use part of our training data (22% in this case) to test the accuracy of our different models.


Testing Different Models
I will be testing the following models with my training data (got the list from here):

Gaussian Naive Bayes
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
KNN or k-Nearest Neighbors
For each model, we set the model, fit it with 80% of our training data, predict for 20% of the training data and check the accuracy.
 Model evaluation_______________
 
                  Model  Score
                  
2        Random Forest  85.79

4        Decision Tree  80.71

1  Logistic Regression  79.19

3          Naive Bayes  78.68

0                  KNN  77.66


[EVERY THING IS EXPLAINED IN THE SOURCE CODE, BETTER READ PROBLEM & DATA UNDERSTANDING FROM THIS README FILE AND RUN THE CODE FOR KNOWING PROCESS CLEARLY WITH OUTPUT]
 
