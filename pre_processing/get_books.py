import os

from pre_processing.Book import Book
from functools import partial
from multiprocessing import Manager, Pool, Value, Array, cpu_count

import pickle
import os.path


NUMBERS_ONLY = True

curdir = os.path.dirname(os.path.realpath(__file__)) + os.sep + ".." + os.sep + "gap-html"
counter = Value('i', 0)


def _get_each_book_files_and_dir():
    """
    gets book folder and the corresponding pages in html ocr format
    :return: list of tuples [(<book_path,[pages]>), (str,[])]
    """
    book_to_files = []
    x = 0
    for subdir, dirs, files in os.walk(curdir):
        pages = []
        for _, _, folder in os.walk(subdir):
            if len(folder) != 0:
                book_path = subdir
                pages.append(folder)

        book_to_files.append((book_path, pages[0]))
    return book_to_files[1:]


def _extract_words_from_book(book_pages, book_list, lock, numbers_only):

    def _add_book_to_list(book_list, book,lock):
        lock.acquire()
        book_list.append(book)
        lock.release()

    with counter.get_lock():
        _id = counter.value
        counter.value += 1
    book = Book(_id, book_pages[0], book_pages[1], numbers_only)
    _add_book_to_list(book_list,book,lock)



def _get_all_books(numbers_only):
    m = Manager()
    l = m.Lock()
    book_list = m.list()
    partial_extract_words_from_book = partial(_extract_words_from_book, book_list=book_list, lock=l, numbers_only=numbers_only)
    books_with_pages = _get_each_book_files_and_dir()
    pool = Pool(processes=cpu_count())
    pool.map(partial_extract_words_from_book, books_with_pages)
    pool.close()
    pool.join()
    return list(book_list)


def get_preprocessed_data(numbers_only=False):
    """
    :return: list of books [<Book>]
    """

    list_of_books = []
    if os.path.exists("pre_processing/preprocessed_books.pickle"):
        with open('pre_processing/preprocessed_books.pickle', 'rb') as handle:
            list_of_books = pickle.load(handle)
    else:
        list_of_books = _get_all_books(numbers_only)
        with open('pre_processing/preprocessed_books.pickle', 'wb') as handle:
            pickle.dump(list_of_books, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return list_of_books

if __name__ == '__main__':
    lb = get_preprocessed_data()
    print("Number of books pre_processed {}".format(len(lb)))
