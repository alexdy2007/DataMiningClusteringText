
from analysis.cosine_similarity import get_cosine_similarity, get_euclidean_distance
from sklearn.manifold import MDS
from sklearn.decomposition import TruncatedSVD

def get_multi_scaling_positions(words_freq_matrix, euclidean=True):
    print(euclidean)
    if not euclidean:
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1) # Using COSINE as dissimilatrity
        dist = get_cosine_similarity(words_freq_matrix)
        pos = mds.fit_transform(dist)
    else:
        mds = MDS(n_components=2, random_state=1) # Using euclidean as dissimilatrity
        dist = get_euclidean_distance(words_freq_matrix)
        pos = mds.fit_transform(dist)
    return pos


def get_LSA_scaling_positions(words_freq_matrix):
    svd = TruncatedSVD(n_components=2)
    return svd.fit_transform(words_freq_matrix)

