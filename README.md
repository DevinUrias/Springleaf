# Springleaf Marketing Response Classification
#### https://www.kaggle.com/competitions/springleaf-marketing-response/overview

## ---Project---

This project tackles the Springleaf Marketing Response Kaggle challenge, where the goal is to predict customer responses (binary classification) using a highly anonymized dataset of over 1800 features and 50,000 records.

-----
## -----Initial Look-----
-----
Rows: 145,232

Features: 1,934 columns (anonymized and mixed-type)

Target: target (0 = no response, 1 = response)

Challenge: Size, anonymized data, categorical fields, columns with multiple datatypes, heavy class imbalance, mixed data entry problems, Test has no target

Cannot do many first step things as the file simply won't load.

### Objectives:
Reduce dimensionality and select meaningful features
Compare and evaluate different classification models.
Optimize model performance using hyperparameter tuning.
Prioritize AUC-ROC.


### Tableau

Insights:
-Many differently formatted binary columns. 1 0, True False, "True" "False"
-Categorical columns like datetime mostly empty
-Heavy skew for target column (~23% positive class)
-Column headers mean nothing
-Target is going to be a binary classification problem

-----
## -----Data Cleaning and Preparation-----
-----

Immediately cut the size of the dataset from 145k to 50k rows. Let me actually open it.

Priority: Reduce columns

On importing many columns gave an error due to mixed datatypes, so isolated them after forcing the import.
-Mostly boolean columns, where saved as both float and bool. Some were just null columns with one or two strings or datetime
-Only 51 columns were categorical total
-Coerced columns which only had boolean like values to be boolean
-Left 2 problematic columns, one datetime one seemed to just be notes
-Dropped columns with over 60% missing values and this handled above issues

Cleanup:
-Dropped all columns which only had one value for all cells
-Dropped columns which were duplicates of others
-looking more specifically at categorical columns still had some boolean columns, others were more in line with what would be expected of a categorical column
-fixed boolean columns

-Replaced null Nan with median for numeric or mode for categorical columns
-Encoded categorical columns
-Finished with still 1845 columns

Feature Selection:
made sure to first filter out unnecessary columns, id and target
-got 2 correlation scores for all columns, 
---one average with regards to all columns
---one with regard to only target
-This gives me 3 datasets I can work with
---Most unique
---Most correlated to target
---"Best of Both"

-----
## -----Machine Learning-----
-----

Planning to run multiple sets of columns I coded a function to run my choice of model and chosen dataset.
Tested 4 models on each dataset:
-RandomForest
-DecisionTree
-XGBoost
-LogisticRegression
These provided a well rounded approach with each model approaching the problem differently

Initial run had decent accuracy when it came to predicting those who would not, but was not effective at determining who WOULD respond.
This is possibly due to the skewed target, overlearning 
Out of first runs the corr dataset did best, with the LogisticRegression model getting .7195
Attempting to use smote to balance the dataset only made it worse somehow
Attempted a grid search and still didn't do better, giving us a final model with AUC-ROC score of .7195


-----
## -----Evaluation-----
-----
Getting additional performance out of this dataset is difficult, and nothing helped improve our model. 
The best result was achieved with LogisticRegression using scaled and imputed data
This gave us our .7195 AUC-ROC score 

### Lessons Learned:
Feature correlation and uniqueness filtering is essential for high-dimensional datasets.
Logistic regression performed surprisingly well with proper scaling and tuning.
SMOTE improved recall for some models but hurt overall AUC.
Without the ability to run anything on the full size dataset this is as far as I can go for now

### Future Work:

Would love to try and play more with what columns I run the machine learning algorithms on, and possibly experiment with even more learning models too. It's curious that SMOTE didn't help at all, so maybe even experimenting with other methods of balancing the skew in the data would be good too. Ultimately, I'm most surprised by how well the regression model did, so if I were able to get a better pc with which to run everything I'd love to see just how well we can do with regards to the full dataset. 
