![UTA-DataScience-Logo](https://github.com/user-attachments/assets/8f615d7f-ee65-4a00-a34e-f376c185c484)

# Springleaf Marketing Response Classification
#### https://www.kaggle.com/competitions/springleaf-marketing-response/overview

## ---Project---

This project focuses on the Springleaf Marketing Response dataset, where the goal is to predict customer responses (binary classification) using a highly anonymized dataset of over 1900 features and 145,000 records. The sheer size of the dataset in addition to the inability to use domain knowledge to help sort features adds a level of challenge one doesn't always experience, and trimming down to the most relevant features quickly becomes the main focus. Through significant reducing of less relevant columns I am able to get a .7195 AUC-ROC score.

-----
## -----Initial Look-----
-----
Rows: 145,232

Features: 1,934 columns (anonymized and mixed-type)

Target: target (0 = no response, 1 = response)

Challenge: Size, anonymized data, columns with multiple datatypes, inconsistent data entry problems, Test file has no target

Cannot do many first step things as the file is simply too big to load.

### Objectives:
Reduce dimensionality and select meaningful features <br>
Compare and evaluate different classification models. <br>
Optimize model performance using hyperparameter tuning. <br>
Prioritize AUC-ROC. <br>


### Tableau Insights:

-Many differently formatted binary columns. 1 0, True False, "True" "False" <br>
-Categorical columns like datetime mostly empty <br>
-Heavy skew for target column (~23% positive class) <br>
-Column headers mean nothing <br>
-Target is going to be a binary classification problem <br>

-----
## -----Data Cleaning and Preparation-----
-----

Immediately cut the size of the dataset from 145k to 50k rows. Let me actually open it.

Priority: Reduce columns

On importing many columns gave an error due to mixed datatypes, so isolated them after forcing the import. <br>
-Mostly boolean columns, where saved as both float and bool. Some were just null columns with one or two strings or datetime <br>
-Only 51 columns were categorical total <br>
-Coerced columns which only had boolean like values to be boolean <br>
-Left 2 problematic columns, one datetime one seemed to just be notes <br>
-Dropped columns with over 60% missing values and this handled above issues <br>

Cleanup: <br>
-Dropped all columns which only had one value for all cells <br>
-Dropped columns which were duplicates of others <br>
-looking more specifically at categorical columns still had some boolean columns, others were more in line with what would be expected of a categorical column <br>
-fixed boolean columns <br>

-Replaced null Nan with median for numeric or mode for categorical columns <br>
-Encoded categorical columns <br>
-Finished with still 1845 columns <br>

Feature Selection: <br>
made sure to first filter out unnecessary columns, id and target <br>
-got 2 correlation scores for all columns,  <br>
---one average with regards to all columns <br>
---one with regard to only target <br>
-This gives me 3 datasets I can work with <br>
---Most unique <br>
---Most correlated to target <br>
---"Best of Both" <br>

-----
## -----Machine Learning-----
-----

Planning to run multiple sets of columns I coded a function to run my choice of model and chosen dataset. <br>
Tested 4 models on each dataset: <br>
-RandomForest <br>
-DecisionTree <br>
-XGBoost <br>
-LogisticRegression <br>
These provided a well rounded approach with each model approaching the problem differently

Initial run had decent accuracy when it came to predicting those who would not, but was not effective at determining who WOULD respond. <br>
This is possibly due to the skewed target, overlearning <br>
Out of first runs the corr dataset did best, with the LogisticRegression model getting .7195 <br>
Attempting to use smote to balance the dataset only made it worse somehow <br>
Attempted a grid search and still didn't do better, giving us a final model with AUC-ROC score of .7195 <br>


-----
## -----Evaluation-----
-----
Getting additional performance out of this dataset is difficult, and nothing helped improve our model. <br>
The best result was achieved with LogisticRegression using scaled and imputed data <br>
This gave us our .7195 AUC-ROC score <br>

### Lessons Learned: <br>
Feature correlation and uniqueness filtering is essential for high-dimensional datasets. <br>
Logistic regression performed surprisingly well with proper scaling and tuning. <br>
SMOTE improved recall for some models but hurt overall AUC. <br>
Without the ability to run anything on the full size dataset this is as far as I can go for now <br>

### Future Work:

Would be interested in trying to experiment more with what columns I run the machine learning algorithms on, and possibly explore more learning models too. It's curious that SMOTE didn't help at all, so maybe even experimenting with other methods of balancing the skew in the data would be good too. Ultimately, I'm most surprised by how well the regression model did, so if I were able to get a better pc with which to run everything I'd love to see just how well we can do with regards to the full dataset. 
