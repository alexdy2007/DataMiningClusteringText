from sklearn.metrics.pairwise import cosine_similarity

def get_cosine_similarity(term_freq_matrix):
    dist = cosine_similarity(term_freq_matrix)
    print("Cosine Distance : {}".format(dist))
    return dist

