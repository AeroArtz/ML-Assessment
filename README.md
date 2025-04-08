# ML-Assessment
Data Analysis and Machine Learning for medical dataset

# 1) Data exploration

age: Patient's age in years
sex: Patient's gender
cp: Chest pain type 
trestbps: Resting blood pressure in mm Hg 
chol: Serum cholesterol in mg/dl 
fbs: Fasting blood sugar > 120 mg/dl
restecg: Resting electrocardiographic results 
thalach: Maximum heart rate achieved 
exang: Exercise induced angina
oldpeak: ST depression induced by exercise relative to rest 
slope: Slope of the peak exercise ST segment
ca: Number of major vessels colored by fluoroscopy
thal: Thalassemia type
target: Presence of heart disease

# splitting columns into discrete and continous 

DISCRETE - sex , cp , fbs , restecg, exang , slope, ca , thal
CONTINOUS - age, trestbps, chol, thalach, oldpeak

# 2) Preprocessing and EDA
After visualizing the distributions of each of the column , it turns out only age is normally distributed and all others are skewed . Some columns have a little skew whereas others are highly skewed with outliers. Furthermore the output classes are also almost equally balanced with a difference of around 2000 samples

Moreover the heatmap indicates that the target variable is correlated to the following variables : cp , thalach , slope and restecg

It can also be seen that columns have varying ranges which have to be scaled/normalized . Since 'age' is normally distributed it can be dealt with a standard scaler whereas the remaining can be dealth with the RobustScaler which is suitable for skewed data which contains outliers as well


# 3) Loading and Training
After preprocessing the data we can split the data ( 80 : 20 ratio for train test split) and train using standard ML algorithms including RandomForestClassifier, DecisionTreeClassifier, KNeighborsClassifier, AdaBoostClassifier, SGDClassifier, GaussianNB, SVC.

In addition a Deep Neural Network was also used with 4 hidden layers (arbitary choice) using ReLu activation function in the hidden layers, and sigmoid in the final layer since its binary classification . As for the loss function binary cross entropy was used, along with adam optimizer


# 3) Model evaluation
For each of these algorithms used the trained model has to be evaluated using a confusion matrix which shows perfomance metrics like Accuracy, Recall, Precision and F1 score.

In order to evaluate the perfomance of the trained neural network a confusion matrix was used along with Accuracy vs epochs diagram

