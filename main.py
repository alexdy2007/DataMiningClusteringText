from pre_processing.get_books import get_preprocessed_data
from pre_processing.frequency_inverse import get_freq_inverse

from analysis.cosine_similarity import get_cosine_similarity
from analysis.k_means import k_means_analysis

list_of_books = get_preprocessed_data()
frequency_term_matrix = get_freq_inverse(list_of_books)
# get_cosine_similarity(frequency_term_matrix)
k_means_analysis(frequency_term_matrix,list_of_books)