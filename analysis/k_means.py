from sklearn.cluster import KMeans
import pandas as pd

def k_means_analysis(text_frequency_analysis, list_of_books):
    num_clusters = 5
    km = KMeans(n_clusters=num_clusters)
    km.fit(text_frequency_analysis)
    clusters = km.labels_.tolist()
    print("HERE")

    titles = []
    published = []
    authors = []
    for book in list_of_books:
        authors.append(book.meta["author"])
        titles.append(book.meta["title"])
        published.append(book.meta["published"])

    books = {'titles': titles, 'cluster': clusters, 'published': published, 'authors':authors}

    frame = pd.DataFrame(books, index = [clusters] , columns = ['titles', 'cluster', 'published','authors']).sort_index()
    print(frame.head(24))
