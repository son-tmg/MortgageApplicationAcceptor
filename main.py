"""
    Tasks :
    
    1- split each column per line into the variables described in the guidelin.

"""

import csv
import matplotlib.pyplot as plt
import numpy as np
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


if __name__ == "__main__":
    applications = read_application('TrainingData.csv')
    # print(len(applications))
    split_point = int(.80*len(applications))
    train_applications = applications[:split_point]
    eval_applications = applications[split_point:]

    ages = {}
    for application in applications:
        age = application[1]
        if age in ages:
            ages[age] += 1
        else:
            ages[age] = 1
        
    ordered_ages = collections.OrderedDict(sorted(ages.items()))

    plt.bar(list(ordered_ages.keys()), ordered_ages.values(), color='g')
    plt.show()