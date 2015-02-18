__author__ = 'ssven'
import csv
import numpy as np

train = csv.reader(open("train.csv", "rU"))
header = train.next()

data = [row for row in train]
data = np.array(data)

# # Original attempt
# women_stats = data[0::,4] == "female"
# men_stats = data[0::,4] != "female"
#
# women_onboard = data[women_stats,1].astype(np.float)
# men_onboard = data[men_stats,1].astype(np.float)
#
# proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)
# proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard)
#
# print "Proportion of women who survived is %s" % proportion_women_survived
# print "Proportion of men who survived is %s" % proportion_men_survived
#
# test_file = open('test.csv', 'rUb')
# test = csv.reader(test_file)
# header = test.next()
#
# prediction_file = open("genderbasedmodel.csv","wb")
# prediction = csv.writer(prediction_file)
#
# prediction.writerow(["PassengerId", "Survived"])
#
# for row in test:
#     prediction.writerow([row[0],'1' if row[3] == 'female' else '0'])
# test_file.close()
# prediction_file.close()

fare_ceiling = 40
data[ data[:,9].astype(np.float) >= fare_ceiling, 9] = fare_ceiling - 1.0

fare_bracket_size = 10
number_of_price_brackets = fare_ceiling / fare_bracket_size

number_of_classes = len(np.unique(data[:,2]))

survival_table = np.zeros((2,number_of_classes,number_of_price_brackets))

for i in xrange(number_of_classes):
    for j in xrange(number_of_price_brackets):
        women_stats = data[(data[:,4] == 'female') & (data[:,2].astype(np.float) == i+1)
                           & (data[:,9].astype(np.float) >= j*fare_bracket_size)
            &(data[:,9].astype(np.float) < (j+1)*fare_bracket_size),1]

        men_stats = data[(data[:,4] != 'female') & (data[:,2].astype(np.float) == i+1)
                           & (data[:,9].astype(np.float) >= j*fare_bracket_size)
            &(data[:,9].astype(np.float) < (j+1)*fare_bracket_size),1]

        survival_table[0,i,j] = np.mean(women_stats.astype(np.float))
        survival_table[1,i,j] = np.mean(men_stats.astype(np.float))

survival_table[survival_table != survival_table ] = 0

print survival_table

survival_table [ survival_table < 0.5]  = 0
survival_table [ survival_table >= 0.5] = 1

print survival_table



test_file = open('test.csv', 'rUb')
test = csv.reader(test_file)
header = test.next()

prediction_file = open("genderbasedmodel.csv","wb")
prediction = csv.writer(prediction_file)

prediction.writerow(["PassengerId", "Survived"])

for row in test:
    for j in xrange(number_of_price_brackets):
        try:
            row[8] = float(row[8])
        except:
            bin_fare = 3 - float(row[1])
            break
        if row[8] > fare_ceiling:
            bin_fare = number_of_price_brackets -1
            break
        if row[8] >= j * fare_bracket_size and row[8] < (j+1)*fare_bracket_size:
            bin_fare = j
            break

    if row[3] == "female":
            prediction.writerow([row[0], "%d" % int(survival_table[0, float(row[1])-1, bin_fare])])
    else:
            prediction.writerow([row[0], "%d" % int(survival_table[1, float(row[1])-1, bin_fare])])


test_file.close()
prediction_file.close()