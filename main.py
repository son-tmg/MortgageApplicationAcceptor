"""
    Tasks :
    
    1- split each column per line into the variables described in the guidelin.

"""

import csv


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
