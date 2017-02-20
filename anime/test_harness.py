import pickle
import pandas as pd
from settings import *

a_file = open(DIR_PROCESSED + '/one_hot_encoded_anime.pickle', 'rb')
anime = pickle.load(a_file)
a_file.close()

# TODO: Deal with nan genre in anime

# "rating" is in both tables
anime["avg_rating"] = anime.rating
anime = anime.drop("rating", 1)

ratings = pd.read_csv(DIR_DATA + '/rating.csv')

shows = pd.merge(anime, ratings)

userRatings = shows.pivot_table(index=['anime_id'], columns='user_id', values='rating')
userRatings.head()
print("Finished creating pivot_table")


a_file = open(DIR_PROCESSED + '/sparse_merged_user_pivot_table.pickle', 'wb')
pickle.dump(userRatings.to_sparse(), a_file)
a_file.close()