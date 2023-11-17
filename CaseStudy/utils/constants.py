import pandas as pd 
import numpy as np
import json
import time
import heapq
import pickle
import matplotlib.pyplot as plt
import time
import bisect 
from sklearn.cluster import KMeans

PROB_STAY = 0.9

def initialize_data(found = True):
    if not found:
        f1 = open('node_data.json')
        nodes_data = json.load(f1)
        
        nodes = {}
        for node in nodes_data:
            nodes[int(node)] = nodes_data[node]

        edges = pd.read_csv('edges.csv')
        adj = {}
        for i in range(len(edges)):
            start_id = int(edges.iloc[i]['start_id'])
            if start_id not in adj:
                adj[start_id] = {}
            end_id = int(edges.iloc[i]['end_id'])
            adj[start_id][end_id] = list(edges.iloc[i][2:])
        
        ps = pd.read_csv('passengers.csv')
        dr = pd.read_csv('drivers.csv')
        ps.columns = ['datetime', 'source_lat', 'source_lon', 'dest_lat', 'dest_lon']
        dr.columns = ['datetime', 'source_lat', 'source_lon']

        ps['datetime'] = pd.to_datetime(ps['datetime'])
        date = pd.to_datetime(ps['datetime'])
        ps['weekday'] = date.dt.dayofweek > 4
        ps['time'] = date.dt.hour
        ps['uid'] = ps.index

        dr['datetime'] = pd.to_datetime(dr['datetime'])
        date = pd.to_datetime(dr['datetime'])
        dr['weekday'] = date.dt.dayofweek > 4
        dr['time'] = date.dt.hour
        dr['did'] = dr.index

        ps['datetime'] = (ps['datetime'].astype(int)/(6 * 1e10)).astype(int)
        dr['datetime'] = (dr['datetime'].astype(int)/(6 * 1e10)).astype(int)
        
        passengers = []
        for p in ps.itertuples():
            row = (p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8])
            passengers.append(row)

        drivers = []
        for d in dr.itertuples():
            row = (d[1], d[2], d[3], d[4], d[5], d[6])
            drivers.append(row)
        
        with open("data/passengers.pkl", "wb") as outfile:
            pickle.dump(passengers, outfile)
        with open("data/drivers.pkl", "wb") as outfile:
            pickle.dump(drivers, outfile)
        with open("data/adj.pkl", "wb") as outfile:
            pickle.dump(adj, outfile)
        with open("data/nodes.pkl", "wb") as outfile:
            pickle.dump(nodes, outfile)
        return
    else:
        f1 = pd.read_pickle(r'data/passengers.pkl')
        f2 = pd.read_pickle(r'data/drivers.pkl')
        f3 = pd.read_pickle(r'data/adj.pkl')
        f4 = pd.read_pickle(r'data/nodes.pkl')
        
        return f1,f2,f3,f4

def get_edge_num(weekday, arrival_time):
    return int(not weekday) * 24 + arrival_time + 1


def initialize_clusters():
    passengers, drivers, adj, nodes = initialize_data(True)
    nodes_df = pd.DataFrame(nodes)
    nodes_df = nodes_df.T.reset_index()
    nodes_df = nodes_df.rename(columns = {'index':'node'})
    plt.rcParams['figure.figsize'] = [7, 5]

    X = nodes_df[['lon', 'lat']]

    NUM_CLUSTERS = 70
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, n_init="auto").fit(X)

    nodes_df['label'] = kmeans.labels_

    for i in range(NUM_CLUSTERS):
        subset = nodes_df.loc[nodes_df['label'] == i]


    centers = pd.DataFrame(kmeans.cluster_centers_)
    centers['label'] = centers.index
    node_centers = {}
    for i in range(NUM_CLUSTERS):
        row = centers.iloc[i]
        node_centers[int(row['label'])] = {'lon':row[0], 'lat':row[1]}

    label_to_centers = {}
    for i in range(NUM_CLUSTERS):
        coord = node_centers[i]
        closest_node = fast_get_closest_node(coord, nodes, threshold = 1, init = True)
        label_to_centers[i] = closest_node

    node_dists = {}
    for i in range(NUM_CLUSTERS):
        node_dists[(i, i)] = 0
        for j in range(i + 1, NUM_CLUSTERS):
            dist = astar(label_to_centers[i], label_to_centers[j], adj, nodes, 19)
            node_dists[(i, j)] = dist
            node_dists[(j, i)] = dist

    nodes_label = {}
    for i in range(len(nodes_df)):
        row = nodes_df.iloc[i]
        nodes_label[int(row['node'])] = int(row['label'])
        
    with open("data/nodes_label.pkl", "wb") as outfile:
        pickle.dump(nodes_label, outfile)
    with open("data/node_dists.pkl", "wb") as outfile:
        pickle.dump(node_dists, outfile)