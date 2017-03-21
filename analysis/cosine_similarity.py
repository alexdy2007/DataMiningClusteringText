from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

def get_cosine_similarity(term_freq_matrix):
    dist = 1 - cosine_similarity(term_freq_matrix)
    return dist

def get_euclidean_distance(term_freq_matrix):
    dist = euclidean_distances(term_freq_matrix)
    return dist
