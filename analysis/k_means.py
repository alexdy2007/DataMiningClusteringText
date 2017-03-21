from sklearn.cluster import KMeans
from analysis.multidimensional_scaling import get_multi_scaling_positions
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


def k_means_analysis(text_frequency_analysis, list_of_books, all_words, terms, drawgraph=True):
    print("Starting K Means")
    num_clusters = 5
    km = KMeans(n_clusters=num_clusters)
    km.fit(text_frequency_analysis)
    clusters = km.labels_.tolist()
    print("Done K Means")

    plt.imshow
    titles = []
    published = []
    authors = []
    period = []
    for book in list_of_books:
        authors.append(book.meta["author"])
        titles.append(book.meta["title"])
        published.append(book.meta["published"])
        period.append(book.meta["period"])

    books = {'titles': titles, 'cluster': clusters, 'published': published, 'authors':authors, 'period':period}

    frame = pd.DataFrame(books, index = [clusters] , columns = ['titles', 'cluster', 'published','authors', 'period']).sort_index()
    frame.to_csv('kmeansresults.csv')

    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    vocab_frame = pd.DataFrame({'words': all_words}, index=all_words)

    cluster_names = {0:[],1:[],2:[],3:[],4:[]}
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

    for i in range(num_clusters):
        #print("Cluster {} words:".format(i))
        for ind in order_centroids[i, :6]:  # replace 6 with n words per cluster
            print('{}'.format(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0], end=','))
            cluster_names[i].append(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0])

    print("DRAWING GRAPH")
    if drawgraph:
        positions = get_multi_scaling_positions(text_frequency_analysis)
        graph_dict = {'x':positions[:, 0], 'y':positions[:, 1], 'label':clusters, 'title':titles}
        df = pd.DataFrame(graph_dict)

        groups = df.groupby('label')

        ig, ax = plt.subplots(figsize=(17, 9))  # set size
        ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

        # iterate through groups to layer the plot
        # note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                    label=cluster_names[name], color=cluster_colors[name],
                    mec='none')
            ax.set_aspect('auto')
            ax.tick_params( \
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
            )
            ax.tick_params( \
                axis='y',  # changes apply to the y-axis
                which='both',  # both major and minor ticks are affected
               )

        ax.legend(numpoints=1)  # show legend with only 1 point


        # add label in x,y position with the label as the film title
        for i in range(len(df)):
            ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)

        plt.show()  # show the plot