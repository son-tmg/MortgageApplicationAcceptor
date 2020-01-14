"""
    Tasks :
    
    1- split each column per line into the variables described in the guideline.
    2- 

"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections


def read_application(document):
    """reads document and extracts all applicant information 

    Arguments:
        document {csv file} -- document containing all applicant entries
    """

    with open(document, newline='') as csvfile:
        filereader = csv.reader(csvfile)
        applications = []
        for row in filereader:
            applications.append(row)
    return applications


def naive_bayes_classifier(identifiers, number_of_columns):
    """Computes the prior probabilities of each identifier given accept/rejec outcome

    Arguments:
        identifier {[type]} -- [description]
        number_of_columns {[type]} -- [description]
    """

    """
        Compute probability of accept
        compute probabilty of reject
        for each identifier
            for each key, value
    """
    total_accept_reject = pd.Series(identifiers[18])

    total_accept = identifiers[18]['true']
    total_reject = identifiers[18]['false']
    print(total_accept_reject)

    prior_probabilities = [None]*number_of_columns

    # for ith_attribute in range(1, 18):
    #     if ith_attribute not in [8, 12, 15, 16]:
    #         total = 0
    #         for value in identifiers[ith_attribute].values():
    #             total += value
    #         for key in identifiers[ith_attribute].keys:


if __name__ == "__main__":
    applications = read_application('TrainingData.csv')
    # print(len(applications))
    split_point = int(.80*len(applications))
    train_applications = applications[:split_point]
    eval_applications = applications[split_point:]

    number_of_columns = len(applications[0])
    # print(number_of_columns)
    # indexes 0 - 18 represent each attribute
    identifiers = [None]*number_of_columns

    for i in range(len(identifiers)):
        new_attribute_dict = {}  # create a new dictionary for each identifier

        for application in applications:
            ith_attribute = application[i]
            if ith_attribute in new_attribute_dict:
                new_attribute_dict[ith_attribute] += 1
            else:
                new_attribute_dict[ith_attribute] = 1

        ordered_dict = collections.OrderedDict(
            sorted(new_attribute_dict.items()))  # sort the dictionary based on key
        # when new dictionary is filled, set it in the identifiery list and move on to next identifier
        identifiers[i] = ordered_dict

    frame = pd.Series(identifiers)
    print(frame)
    # for ith_attribute in range(1, 18):
    #     if ith_attribute not in [8, 12, 15, 16]:
    #         plt.bar(list(identifiers[ith_attribute].keys()),
    #                 identifiers[ith_attribute].values(), color='g')
    #         plt.show()

    naive_bayes_classifier(identifiers, number_of_columns)
