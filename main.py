from pre_processing.get_books import get_preprocessed_data
from pre_processing.frequency_inverse import get_freq_inverse
from pre_processing.get_all_words import get_all_words

from analysis.k_means import k_means_analysis

NUMBERS_ONLY = False


list_of_books = get_preprocessed_data(NUMBERS_ONLY)
all_words_list = get_all_words(list_of_books)
frequency_term_matrix, terms= get_freq_inverse(list_of_books)
k_means_analysis(frequency_term_matrix,list_of_books, all_words_list, terms)
