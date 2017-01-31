import numpy as np
from math import sqrt


# Distance-based simularity score
def sim_dist(prefs, person1, person2):
    # Get list of shared items
    p1_set = set(prefs[person1].keys())
    p2_set = set(prefs[person2].keys())

    shared = p1_set & p2_set

    sum_of_squares = 0

    for item in shared:
        sum_of_squares += (prefs[person1][item] - prefs[person2][item])**2

    return 1 / (1 + sqrt(sum_of_squares))

# Returns Pearson Correlation coefficient for p1 and p2
def sim_pearson(prefs, p1, p2):
    # Get list of shared items
    p1_set = set(prefs[p1].keys())
    p2_set = set(prefs[p2].keys())

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

# Returns best matches for person from prefs dictionary
def topMatches(prefs, person, n=5, similarity=sim_pearson):
    scores = [(similarity(prefs, person, other), other)
              for other in prefs if other != person]

    scores.sort(reverse=True)
    return scores[0:n]