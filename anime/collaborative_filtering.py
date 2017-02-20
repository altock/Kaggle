import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from settings import *
from math import sqrt

import random
# import xgboost as xgb
from sklearn import preprocessing, neighbors, svm
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
  RandomForestRegressor, AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression


from recommendation import *

# Anime.csv

# anime_id - myanimelist.net's unique id identifying an anime.
# name - full name of anime.
# genre - comma separated list of genres for this anime.
# type - movie, TV, OVA, etc.
# rating - average rating out of 10 for this anime.
# members - number of community members that are in this anime's "group".


# Rating.csv

# user_id - non identifiable randomly generated user id.
# anime_id - the anime that this user has rated.
# rating - rating out of 10 this user has assigned (-1 if the user watched it but didn't assign a rating).


def get_user_preferences(ratings):
    user_preferences = defaultdict(dict)

    for _, row in ratings.iterrows():
        user = row["user_id"]
        anime_id = row["anime_id"]
        rating = row["rating"]

        user_preferences[user][anime_id] = rating

    return user_preferences

def pickle_user_preferences(user_preferences):
    with open(DIR_PROCESSED + '/user_preferences_dict.pickle', 'wb') as p_file:
        pickle.dump(user_preferences, p_file)


# Predict ratings within animes
def predict_on_anime(animes, clf, test_size=0.2, rating='rating'):
    X = animes.drop(['anime_id', 'name', 'rating', "episodes", "int_rating"], 1)
    y = animes[rating]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)


    clf.fit(X_train, y_train)
    print(rating)
    print('Training Score: {:.3f}'.format(clf.score(X_train, y_train)))
    print('Testing Score: %.3f' % (clf.score(X_test, y_test)))
    print()

def predict_collaborative(user_preferences, similarity=sim_pearson):
    # Mean squared error, leave one out
    sqrd_error = 0

    abs_error = 0
    num_scored = 0



    for user, prefs in user_preferences.items():
        rated_animes = [key for key, rating in prefs.items() if rating != -1]
        if len(rated_animes) < 2:
            # No point rating people who haven't watched much
            # TODO: Full coverage later
            continue

        anime_id = random.choice(rated_animes)
        rating = user_preferences[user][anime_id]
        estimate = get_reccomendation(anime_id, user_preferences, user, similarity=similarity)
        if estimate is None or estimate >= 10.5:
            continue
        sqrd_error += (rating - estimate) ** 2
        abs_error += abs(rating-estimate)
        print('\t', rating, estimate)
        num_scored += 1

        if num_scored == 0:
            continue
        print(user, sqrd_error/num_scored, sqrt(sqrd_error / num_scored), abs_error/num_scored)

    return sqrd_error / num_scored





if __name__ == '__main__':
    # a_file = open(DIR_PROCESSED + '/one_hot_encoded_anime.pickle', 'rb')
    # anime = pickle.load(a_file)
    # a_file.close()
    #
    # clf = KNeighborsClassifier(n_jobs=-1)
    # predict_on_anime(anime, clf, rating='int_rating')
    #
    #
    # clf = KNeighborsRegressor(n_jobs=-1)
    # predict_on_anime(anime, clf)

    # TODO: Testing system to take test show out of collaboritive filtering
    # TODO: average two scores



    # ratings = pd.read_csv(DIR_DATA + '/rating.csv')
    # labeled_ratings = ratings
    # user_preferences = get_user_preferences(ratings)
    # pickle_user_preferences(user_preferences)
    # print("Pickled")

    with open(DIR_PROCESSED + '/user_preferences_dict.pickle', 'rb') as p_file:
        user_preferences = pickle.load(p_file)
        predict_collaborative(user_preferences, similarity=sim_jaccard)
    #     users = list(user_preferences.keys())
    #
    #     # print(type(user_preferences[users[0]][str(11266)]))
    #
    #     p3 = users[2]
    #     # print(user_preferences[p1])
    #     # for p2 in users[1:1000]:
    #     #     print(sim_pearson(user_preferences, p1, p2))
    #     # print(get_recommendations(user_preferences, p3) similarity=sim_dist))
    #     print(get_recommendations(user_preferences, p3, similarity=sim_jaccard))
