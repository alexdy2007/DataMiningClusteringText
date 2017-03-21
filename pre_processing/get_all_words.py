import nltk
import re

def _tokenize_words(text):
    words = [word.lower() for sentence in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sentence)]
    filtered_tokens = []
    for word in words:
        filtered_tokens.append(word)
    return filtered_tokens

def get_all_words(book_list):
    book_words_list = []
    total_vocab = []
    for book in book_list:
        book_words_list.append(" ".join(book.word_list))

    for book_words in book_words_list:
        words_tokenised = _tokenize_words(book_words)
        total_vocab.extend(words_tokenised)
    print("Done getting word list")
    return total_vocab

