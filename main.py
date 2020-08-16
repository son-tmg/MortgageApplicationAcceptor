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
dftrain = dftrain.replace(np.nan, '', regex=True)
dftrain = dftrain.drop("id", axis=1)

dfeval = pd.read_csv('ForEvaluation.csv',
                     names=name_headers_train, header=0)
dfeval = dfeval.replace(np.nan, '', regex=True)
dfeval = dfeval.drop('id', axis=1)

for column in dftrain:
    if dftrain[column].dtype == 'bool':
        dftrain[column] = dftrain[column].apply(
            lambda x: 1 if x == True else 0)
        if(column != 'Approved'):
            dfeval[column] = dfeval[column].apply(
                lambda x: 1 if x == True else 0)
    elif dftrain[column].dtype == 'object':
        print(column)
        dftrain[column].astype(str)
        if(column != 'Approved'):
            dfeval[column] = dfeval[column].astype(str)


y_train = dftrain
y_eval = dfeval

# %%
# dftrain[2].value_counts().plot(kind='bar')
# dftrain['age'].value_counts().plot(kind='bar')
dftrain.head()
# dfeval.head()
# dftrain.describe()
# dftrain.shape[0]
# dftrain.shape[1]
# dftrain.IsMarried.value_counts().hist()

# %%

numerical_column = []
categorical_column = []
feature_columns = []

for column in dftrain:
    print(column, ': ', dftrain[column].dtype)
    if dftrain[column].dtype == 'int64':
        numerical_column.append(column)
    else:
        categorical_column.append(column)

print(numerical_column)
print(categorical_column)

# %%
for column in categorical_column:
    unique_values = dftrain[column].unique()
    feature_columns.append(
        tf.feature_column.categorical_column_with_vocabulary_list(
            column, unique_values))

for feature in numerical_column:
    feature_columns.append(
        tf.feature_column.numeric_column(feature, dtype=tf.int64))

# %%

# create input function


def make_input_fn(data_df, labels, num_epochs=1, shuffle=True, batch_size=32):
    def input_function():
        dataset = tf.data.Dataset.from_tensor_slices((dict(data_df), labels))
        if shuffle:
            dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size).repeat(num_epochs)
        return dataset
    return input_function


# %%
# create and train model
train_function = make_input_fn(dftrain, y_train)
eval_train_function = make_input_fn(
    dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_function)
result = linear_est.evaluate(eval_train_function)

clear_output()
print('accuracy: ', result)

# %%
