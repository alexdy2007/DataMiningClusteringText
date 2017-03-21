from sklearn.feature_extraction.text import TfidfVectorizer


def get_freq_inverse(book_list, max=0.8, min=0.2, ngram=3, max_features=200000):
    # define vectorizer parameters
    book_words_list= []
    for book in book_list:
        book_words_list.append(" ".join(book.word_list))

    term_freq_vectorizer = TfidfVectorizer(max_df=max, max_features=200000,
                                       min_df=min, stop_words='english',
                                       use_idf=True, ngram_range=(1, ngram))

    term_freq_matrix = term_freq_vectorizer.fit_transform(book_words_list)
    print(term_freq_matrix.shape)
    terms = term_freq_vectorizer.get_feature_names()
    return term_freq_matrix, terms, term_freq_vectorizer