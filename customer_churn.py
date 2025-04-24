#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



#loading the dataset
df = pd.read_csv(r'C:\Users\Hp\Downloads\churn.csv')
df.head()

#checking the shape of the dataset
df.shape

#drop coulumns
df = df.drop(['RowNumber','CustomerId','Surname'], axis=1)

df.isnull().sum()

#column data types
df.dtypes

#dulicate values
df.duplicated().sum()

#rename column
df.rename(columns={'Exited':'Churn'}, inplace=True)

#descriptive statistics
df.describe()

df.head()

#pie chart
plt.figure(figsize=(10,6))
plt.pie(df['Churn'].value_counts(),labels=['No','Yes'],autopct='%1.2f%%')
plt.title('Churn Percentage')
plt.show()

#gender and customer churn
sns.countplot(x = 'Gender', data = df, hue = 'Churn')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

#histogram for age distribution
sns.histplot(data=df, x="Age", hue="Churn", multiple="stack",kde=True)

fig, ax = plt.subplots(1,2,figsize=(15, 5))
sns.boxplot(x="Churn", y="CreditScore", data=df, ax=ax[0])
sns.violinplot(x="Churn", y="CreditScore", data=df, ax=ax[1])

sns.countplot(x = 'Geography', hue = 'Churn', data = df)
plt.title('Geography and Churn')
plt.xlabel('Geography')
plt.ylabel('Count')
plt.show()

fig,ax = plt.subplots(1,2,figsize=(15,5))
sns.countplot(x='Tenure', data=df,ax=ax[0])
sns.countplot(x='Tenure', hue='Churn', data=df,ax=ax[1])
plt.show()

sns.histplot(data=df, x="Balance", hue="Churn", multiple="stack",kde=True)
plt.show()

sns.countplot(x='NumOfProducts', hue='Churn', data=df)
plt.show()

sns.countplot(x=df['HasCrCard'],hue=df['Churn'])

sns.countplot(x='IsActiveMember', hue='Churn', data=df)
plt.show()

sns.histplot(data=df,x='EstimatedSalary',hue='Churn',multiple='stack',palette='Set2')
plt.show()


#label encoding
variables = ['Geography','Gender']
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in variables:
    le.fit(df[i].unique())
    df[i]=le.transform(df[i])
    print(i,df[i].unique())


#normalize the continuous variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['CreditScore','Balance','EstimatedSalary']] = scaler.fit_transform(df[['CreditScore','Balance','EstimatedSalary']])

plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df.drop('Churn',axis=1),df['Churn'],test_size=0.3,random_state=42)

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# Set up MLflow experiment
mlflow.set_experiment("Customer Churn Prediction")

models = {
    "DecisionTree": DecisionTreeClassifier(),
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

param_grids = {
    "DecisionTree": {
        'max_depth': [2, 4, 6, 8, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        'criterion': ['gini', 'entropy'],
        'random_state': [0, 42]
    },
    "LogisticRegression": {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    },
    "RandomForest": {
        'n_estimators': [50, 100, 150],
        'max_depth': [2, 4, 6, 8],
        'min_samples_split': [2, 5, 10],
        'random_state': [0, 42]
    },
    "XGBoost": {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9]
    }
}

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Make predictions
        y_pred = best_model.predict(X_test)

        # Compute evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics to MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log confusion matrix as an artifact
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {model_name}')
        plt.savefig(f'confusion_matrix_{model_name}.png')
        mlflow.log_artifact(f'confusion_matrix_{model_name}.png')


import seaborn as sns
import matplotlib.pyplot as plt

for model_name, model in models.items():
    # Predict using the best model from GridSearchCV
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Plot actual vs. predicted distributions
    plt.figure(figsize=(8, 6))
    ax = sns.kdeplot(y_test, color="r", label="Actual Values", linewidth=2)
    sns.kdeplot(y_pred, color="b", label=f"Predicted ({model_name})", ax=ax, linewidth=2)

    plt.title(f"Actual vs. Predicted Distribution ({model_name})")
    plt.xlabel("Class")
    plt.ylabel("Density")
    plt.legend()
 
    
    # Show plot
    plt.show()


from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score

for model_name, model in models.items():
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Compute ROC-AUC
    try:
        y_prob = best_model.predict_proba(X_test)[:, 1]  # If model supports probability predictions
        roc_auc = roc_auc_score(y_test, y_prob)
    except:
        roc_auc = roc_auc_score(y_test, y_pred)  # If no probability output, use direct predictions

    print(f"Model: {model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc)
    print("-" * 30)  # Separator for readability

