import csv
import numpy as np


def train():
    # Get training data
    file = csv.reader(open('../data/train.csv', 'rbU'))

    # first line isn't data
    file.next()

    data = np.array([row for row in file])
    return data


def test():
    file = csv.reader(open('../data/test.csv', 'rbU'))
    file.next()

    pred_file = csv.writer(open('../data/genderbasedmodel.csv', 'wb'))

    pred_file.writerow(["PassengerId", "Survived"])
    for row in file:
        if row[3] == 'female':
            pred_file.writerow([row[0],  '1'])
        else:
            pred_file.writerow([row[0], '0'])


if __name__ == "__main__":
    data = train()
