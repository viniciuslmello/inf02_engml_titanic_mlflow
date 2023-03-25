#!/bin/env/python
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import typer

def load_data():
    data = pd.read_csv('../data/01_raw/train.csv')
    data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    return data

def conform_data(data):
    conformed = (
        pd.merge(
            data,
            pd.get_dummies(data['Embarked'], prefix='Embarked'),
            left_index=True, right_index=True
        )
        .drop(columns=['Embarked'])
        .assign(Sex=lambda df: df['Sex'] == 'female')
    ).astype('float64')
    conformed.dropna(inplace=True)
    return conformed

def train_logistic_regression(data, n_folds):
    X = data.drop(columns = ['Survived']).copy()
    y = data['Survived']
    
    params = {
        'penalty': ['l1', 'l2'],
        'C': [2, 10, 20],
    }
    
    cv = StratifiedKFold(n_splits=n_folds)
    
    model_template = LogisticRegression(solver='saga', max_iter=10000)

    clf = GridSearchCV(
        model_template,
        params,
        cv=cv,
        scoring=['f1','precision','recall'],
        refit='f1',
        return_train_score=True,
    )
    
    clf.fit(X,y)
    
    return clf
    
def report_model(clf):
    idx = clf.best_index_
    
    print(f"Melhor Score de F1: {clf.best_score_}")
    print(f"Melhor Parametro: {clf.best_params_}")
    
    print(f"Melhor F1 médio: {clf.cv_results_['mean_test_f1'][idx]}")
    mlflow.log_metric('f1_mean',clf.cv_results_['mean_test_f1'][idx])
    
    print(f"Melhor F1 desvio: {clf.cv_results_['std_test_f1'][idx]}")
    mlflow.log_metric('f1_std',clf.cv_results_['std_test_f1'][idx])
    
    print(f"Melhor Precision médio: {clf.cv_results_['mean_test_precision'][idx]}")
    mlflow.log_metric('Precision_mean',clf.cv_results_['mean_test_precision'][idx])
    
    print(f"Melhor Precision desvio: {clf.cv_results_['std_test_precision'][idx]}")    
    mlflow.log_metric('Precision_std',clf.cv_results_['std_test_precision'][idx])
    
    print(f"Melhor Recall médio: {clf.cv_results_['mean_test_recall'][idx]}")
    mlflow.log_metric('Recall_mean',clf.cv_results_['mean_test_recall'][idx])
    
    print(f"Melhor Recall desvio: {clf.cv_results_['std_test_recall'][idx]}")
    mlflow.log_metric('Recall_std',clf.cv_results_['std_test_recall'][idx])
        
    print(f"Resultado da Validação Cruzada")
    print(clf.cv_results_)

#################################################
# Linha de Comando
#################################################    

# Tentativa de usar o click que não funcionou
# import click

# @click.command()
# def hello():
#     print("Hello World")
    

app = typer.Typer()

@app.command()
def train_lr(n_folds : int = 10):
    print(f"Executando Validação Cruzada com k={n_folds}")
    
    experiment_id = mlflow.create_experiment('classificador_titanic')
    
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param('model', 'logistic_regression')
        mlflow.log_param('normaalization', 'none')
        mlflow.log_param('n_folds', n_folds)
        
        data = load_data()
        data = conform_data(data)
        clf = train_logistic_regression(data, n_folds)
        report_model(clf)
    
if __name__  ==  "__main__":
    app()



