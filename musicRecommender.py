# Anita Soroush

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os
import csv

def music_recommender(userPreferences):
        raw_data = pd.read_csv('genres_v2.csv', dtype={'song_name': 'str'})
        print(raw_data.shape)
        pd.set_option('display.max_columns', None)
        raw_data.info()

        # data cleaning --------------------------------------------------------------------------------------
        nulls = raw_data.isnull().sum()
        print(nulls)

        training_data = raw_data.drop(['type', 'uri', 'track_href', 'analysis_url',
                                       'song_name', 'Unnamed: 0', 'title', 'genre'], axis=1, inplace=False)

        print(training_data.shape)
        training_data = training_data[training_data.key != -1]
        print("after dropping some rows:\n", training_data.shape)
        print(training_data.head())

        print(training_data.shape)
        print(training_data.duplicated().any())
        training_data.drop_duplicates(inplace=True)
        print(training_data.shape)

        training_data.hist()
        plt.show()

        # this global scalar will be fitted on training data and will be used for both training and test data
        global_scalar = MinMaxScaler()
        id_column = training_data['id']
        training_data.drop(['id'], axis=1, inplace=True)
        global_scalar.fit(training_data)
        training_data = pd.DataFrame(global_scalar.transform(training_data),
                                     index=training_data.index,
                                     columns=training_data.columns)
        training_data['id'] = id_column

        training_data.hist()
        plt.show()

        training_data.info()

        corr = training_data.corr()
        sns.heatmap(corr[corr > 0.1], cmap="Blues", annot=True)
        plt.show()

        # clustering ----------------------------------------------------------------------------------------

        wcss = []
        for i in range(1, 20):
            kmeans = KMeans(i)
            kmeans.fit(training_data.drop(['id'], axis=1, inplace=False))
            wcss_iter = kmeans.inertia_
            wcss.append(wcss_iter)

        number_clusters = range(1, 20)
        plt.plot(number_clusters, wcss)
        plt.title('The Elbow title')
        plt.xlabel('Number of clusters')
        plt.ylabel('SSE')
        plt.show()

        kmeans = KMeans(n_clusters=10)
        training_data_clustered = kmeans.fit(training_data.drop(['id'], axis=1, inplace=False))
        training_data["cluster"] = training_data_clustered.labels_
        centroids = training_data_clustered.cluster_centers_
        print(training_data.head())

        # making output...........................................................................................
        userPreferences.drop(userPreferences.columns.difference(["danceability", "energy", "key", "loudness", "mode",
                                                                 "speechiness", "acousticness", "instrumentalness",
                                                                 "liveness", "valence", "tempo", "duration_ms",
                                                                 "time_signature"]), 1, inplace=True)

        # input normalizing
        userPreferences = pd.DataFrame(global_scalar.transform(userPreferences),
                                       index=userPreferences.index,
                                       columns=userPreferences.columns)

        fields = ["id", "cluster"]

        # single playlist
        single_playlist = []
        for i in range(5):
            cluster_index = (training_data_clustered.predict(userPreferences.iloc[[i]]))[0]
            print(cluster_index)
            cluster_songs = training_data[training_data.cluster == cluster_index]
            cluster_songs.drop(cluster_songs.columns.difference(["id", "cluster"]), 1, inplace=True)
            single_playlist.append((cluster_songs.sample()).values.flatten().tolist())
            print(single_playlist[i])

        filename = "single_playlist.csv"

        # writing to csv file
        with open(filename, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the fields
            csvwriter.writerow(fields)

            # writing the data rows
            csvwriter.writerows(single_playlist)

        # 5 playlists
        for i in range(5):
            ith_playlist = []
            filename = "pl" + str(i + 1) + ".csv"
            cluster_index = (training_data_clustered.predict(userPreferences.iloc[[i]]))[0]
            cluster_songs = training_data[training_data.cluster == cluster_index]
            cluster_songs.drop(cluster_songs.columns.difference(["id", "cluster"]), 1, inplace=True)
            for j in range(5):
                ith_playlist.append((cluster_songs.sample()).values.flatten().tolist())

            with open(filename, 'w') as csvfile:
                # creating a csv writer object
                csvwriter = csv.writer(csvfile)

                # writing the fields
                csvwriter.writerow(fields)

                # writing the data rows
                csvwriter.writerows(ith_playlist)


def main(args) -> None:
    """ Main function to be called when the script is run from the command line. 
    This function will recommend songs based on the user's input and save the
    playlist to a csv file.
    
    Parameters
    ----------
    args: list 
        list of arguments from the command line (here is just the path of a file like input_tracks.csv)
    Returns
    -------
    None
    """
    arg_list = args[1:]
    if len(arg_list) == 0:
        print("Usage: python3 musicRecommender.py <csv file>")
        sys.exit()
    else:
        file_name = arg_list[0]
        if not os.path.isfile(file_name):
            print("File does not exist")
            sys.exit()
        else:
            userPreferences = pd.read_csv(file_name)
            music_recommender(userPreferences)

if __name__ == "__main__":
    """get arguments from command line
    you just have to write the name of the file that contains the users favorite tracks.
    these tracks are now in input_tracks.csv """
    args = sys.argv
    main(args)