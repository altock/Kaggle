import pickle
import pandas as pd
import numpy as np

from settings import *
from math import sqrt

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
    user_preferences = {}
    for _, row in ratings.iterrows():
        user = row["user_id"]
        anime_id = row["anime_id"]
        rating = row["rating"]

        if user in user_preferences:
            user_preferences[user][anime_id] = rating
        else:
            user_preferences[user] = {anime_id : rating}
    return user_preferences

def pickle_user_preferences(user_preferences):
    with open(DIR_PROCESSED + '/user_preferences_dict.pickle', 'wb') as p_file:
        pickle.dump(user_preferences, p_file)


if __name__ == '__main__':
    # a_file = open(DIR_PROCESSED + '/one_hot_encoded_anime.pickle', 'rb')
    # anime = pickle.load(a_file)
    # a_file.close()

    # ratings = pd.read_csv(DIR_DATA + '/rating.csv')
    # labeled_ratings = ratings[ratings["rating"] != -1]
    # user_preferences = get_user_preferences(ratings)
    # pickle_user_preferences(user_preferences)

    with open(DIR_PROCESSED + '/user_preferences_dict.pickle', 'rb') as p_file:
        user_preferences = pickle.load(p_file)
        users = list(user_preferences.keys())

        # print(type(user_preferences[users[0]][str(11266)]))

        p1 = users[0]
        # for p2 in users[1:1000]:
        #     print(sim_pearson(user_preferences, p1, p2))
        print(topMatches(user_preferences, p1))
