import numpy as np
from math import sqrt
from collections import defaultdict

# Distance-based simularity score
def sim_dist(prefs, person1, person2, show=None):
    # Get list of shared items
    p1_set = set([key for key, value in list(prefs[person1].items()) if value != -1 and show != key])
    p2_set = set([key for key, value in list(prefs[person2].items()) if value != -1 and show != key])

    shared = p1_set & p2_set

    sum_of_squares = 0

    for item in shared:
        sum_of_squares += (prefs[person1][item] - prefs[person2][item])**2

    return 1 / (1 + sqrt(sum_of_squares))

# Returns Pearson Correlation coefficient for p1 and p2
def sim_pearson(prefs, p1, p2, show=None):
    # Get list of shared items
    p1_set = set([key for key, value in list(prefs[p1].items()) if value != -1 and show != key])
    p2_set = set([key for key, value in list(prefs[p2].items()) if value != -1 and show != key])

    shared = list(p1_set & p2_set)

    n = len(shared)

    if n == 0:
        return 0

    p1_prefs = np.array([prefs[p1][item] for item in shared])
    p2_prefs = np.array([prefs[p2][item] for item in shared])

    # E[X], E[Y]
    sum1 = sum(p1_prefs)
    sum2 = sum(p2_prefs)

    # E[X**2], E[Y**2]
    sum1Sq = np.dot(p1_prefs, p1_prefs)
    sum2sq = np.dot(p2_prefs, p2_prefs)

    # E[XY]
    pSum = np.dot(p1_prefs, p2_prefs)

    # Calculate Pearson score

    # numerator = covariance = E[XY] - E[X]E[Y] / n
    num = pSum - (sum1 * sum2 / n)

    # Std_x = sqrt(E[X**2] - E[X]**2 / n)
    # denominator = std_x * std_y
    den = sqrt((sum1Sq - pow(sum1, 2)/n) * (sum2sq - pow(sum2, 2)/n))
    if den == 0:
        return 0

    r = num / den
    return r

# Jaccard index
def sim_jaccard(prefs, p1, p2, show=None):
    # Get list of shared items
    p1_set = set([key for key, value in list(prefs[p1].items()) if value != -1 and show != key])
    p2_set = set([key for key, value in list(prefs[p2].items()) if value != -1 and show != key])

    shared = p1_set & p2_set
    n = len(shared)
    a = len(p1_set)
    b = len(p2_set)

    if a == b == 0:
        return 1
    return n / (a + b - n)


# Returns best matches for person from prefs dictionary
def top_matches(prefs, person, n=5, similarity=sim_pearson):
    scores = [(similarity(prefs, person, other), other)
              for other in prefs if other != person]

    scores.sort(reverse=True)
    return scores[0:n]

def get_reccomendation(show, prefs, person, similarity=sim_pearson):
    total = 0
    simSum = 0

    for other in prefs:
        # Don't compare with yourself
        if other == person or show not in prefs[other] or prefs[other][show] == -1:
            continue

        sim = similarity(prefs, person, other, show)

        # if sim <= 0:
        #     continue
        # Similarity * score
        total += prefs[other][show] * sim

        # sum of similarities
        simSum += sim


    # Normalize list
    if simSum != 0:
        return total/simSum
    else:
        return None

# Get's recommendations using weighted average of other user's rankings
def get_recommendations(prefs, person, similarity=sim_pearson):
    totals = defaultdict(int)
    simSums = defaultdict(int)

    for other in prefs:
        # Don't compare with yourself
        if other == person:
            continue

        sim = similarity(prefs, person, other)

        # if sim <= 0:
        #     continue

        for item in prefs[other]:
            # Only score movies I haven't seen yet
            if (item not in prefs[person] or prefs[person][item] == -1) and not \
                            prefs[other][item] == -1:
                # Similarity * score
                totals[item] += prefs[other][item] * sim

                # sum of similarities
                simSums[item] += sim

    # Normalize list
    rankings = [(total/simSums[item], item) for item, total in list(totals.items()) if simSums[item] != 0]

    return sorted(rankings, reverse=True)