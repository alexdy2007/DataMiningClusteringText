from sklearn.feature_extraction.text import TfidfVectorizer


def get_freq_inverse(book_list):
    # define vectorizer parameters
    book_words_list= []
    for book in book_list:
        book_words_list.append(" ".join(book.word_list))

    term_freq_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                       min_df=0.2, stop_words='english',
                                       use_idf=True, ngram_range=(1, 3))

    term_freq_matrix = term_freq_vectorizer.fit_transform(book_words_list)
    print(term_freq_matrix.shape)
    terms = term_freq_vectorizer.get_feature_names()
    print("terms : {}".format(terms))
    return term_freq_matrix