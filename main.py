# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

name_headers = ['id', 'age', 'CurrentResidenceYears', 'IsMarried', 'NumberOfDependants', 'Graduated', 'SelfEmployed', 'YearsOfJobStability', 'YearlySalary', 'CreditRating', 'CoApplicantAge',
                'CoApplicantYearsOfJobStability', 'CoApplicantYearlySalary', 'CoApplicantCreditRating', 'LoanTermInYears', 'LoanAmount', 'PropertyTotalCost', 'AreaClassification', 'Approved']

name_headers_train = name_headers[:-1]

dftrain = pd.read_csv('trainingData.csv',
                      names=name_headers, header=0)

dfeval = pd.read_csv('ForEvaluation.csv',
                     names=name_headers_train, header=0)

# drop uneccessary column
dftrain = dftrain.drop(columns=['id', 'CurrentResidenceYears', 'IsMarried', 'NumberOfDependants', 'Graduated', 'SelfEmployed', 'YearsOfJobStability', 'YearlySalary', 'CreditRating', 'CoApplicantAge',
                                'CoApplicantYearsOfJobStability', 'CoApplicantYearlySalary', 'CoApplicantCreditRating', 'LoanTermInYears', 'LoanAmount', 'PropertyTotalCost', 'AreaClassification'], axis=1)

dfeval = dfeval.drop(columns=['id',  'CurrentResidenceYears', 'IsMarried', 'NumberOfDependants', 'Graduated', 'SelfEmployed', 'YearsOfJobStability', 'YearlySalary', 'CreditRating', 'CoApplicantAge',
                              'CoApplicantYearsOfJobStability', 'CoApplicantYearlySalary', 'CoApplicantCreditRating', 'LoanTermInYears', 'LoanAmount', 'PropertyTotalCost', 'AreaClassification'], axis=1)

# replace nulls with empty string
dftrain = dftrain.replace(np.nan, '', regex=True)
dfeval = dfeval.replace(np.nan, '', regex=True)

# set booleans to 0 or 1
for column in dftrain:
    if dftrain[column].dtype == 'bool':
        dftrain[column] = dftrain[column].apply(
            lambda x: 1 if x == True else 0)
        if(column != 'Approved'):
            dfeval[column] = dfeval[column].apply(
                lambda x: 1 if x == True else 0)

# get training labels
y_train = dftrain.pop('Approved')
y_eval = dfeval

# %%
# dftrain[2].value_counts().plot(kind='bar')
print('train')
dftrain.describe()

# %%

print('eval')
dfeval.describe()
# %%
# find numerical and categorical feature columns

numerical_column = []
categorical_column = []
feature_columns = []

# put collumns into right category of column
for column in dftrain:
    if column != 'Approved':
        # print(column, ': ', dftrain[column].dtype)
        if dftrain[column].dtype == 'int64':
            numerical_column.append(column)
        else:
            categorical_column.append(column)


# create feature columns
for feature in categorical_column:
    unique_values = dftrain[feature].unique()
    feature_columns.append(
        tf.feature_column.categorical_column_with_vocabulary_list(
            feature, unique_values))

for feature in numerical_column:
    feature_columns.append(
        tf.feature_column.numeric_column(feature, dtype=tf.int64))


# %%
# create input function to convert data to tf.data.Dataset

def make_input_fn(data_df, labels, num_epochs=1, shuffle=True, batch_size=32):
    def input_function():
        dataset = tf.data.Dataset.from_tensor_slices((dict(data_df), labels))
        if shuffle:
            dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size).repeat(num_epochs)
        return dataset
    return input_function

# training and eval functions


train_function = make_input_fn(dftrain, y_train)

eval_train_function = make_input_fn(
    dfeval, y_eval, num_epochs=1, shuffle=False)


# %%
# create and train model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_function)


# get result
# result = linear_est.evaluate(eval_train_function)

# clear_output()
# print('result: ', result)
# wont work, since the provided evaluation set does not have approved column for us to verify accuracy


# %%

pred_dicts = list(linear_est.predict(eval_train_function))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')

# %%
