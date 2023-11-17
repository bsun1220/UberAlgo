from utils.constants import *
from math import sqrt

def dijkstra(start, target, adj, nodes, cap, init = True):
    dist, pq = {}, []
    if init:
        dist, pq = pd.read_pickle('data/init_search.pkl')
        dist[start] = 0
        heapq.heappush(pq, (dist[start], start))
    else:
        for node in nodes:
            dist[node] = float('inf')
        dist[start] = 0
        for node in dist:
            heapq.heappush(pq, (dist[node], node))
    
    
    while len(pq):
        distance, node = heapq.heappop(pq)
        
        for neighbor in adj[node]:
            
            times = distance + adj[node][neighbor][0]/adj[node][neighbor][cap]
            if dist[neighbor] > times:
                dist[neighbor] = times
                heapq.heappush(pq, (dist[neighbor], neighbor))
            
            if neighbor == target:
                return (dist[neighbor] * 60)
    
    return -1

def astar(start, target, adj, nodes, cap, init = True):
    dist, pq = {}, []
    if init:
        dist, pq = pd.read_pickle('data/init_search.pkl')
        dist[start] = 0
        heapq.heappush(pq, (dist[start], start))
    else:
        for node in nodes:
            dist[node] = float('inf')
        dist[start] = 0
        for node in dist:
            heapq.heappush(pq, (dist[node], node))
    
    dist[start] = 0
    while len(pq):
        f_best, node = heapq.heappop(pq)
        distance = dist[node]
        
        for neighbor in adj[node]:
            
            times = distance + adj[node][neighbor][0]/adj[node][neighbor][cap]
            lon_dist = (nodes[target]['lon'] - nodes[neighbor]['lon'])
            lat_dist = (nodes[target]['lat'] - nodes[neighbor]['lat'])
            
            euclid = sqrt(lon_dist * lon_dist + lat_dist * lat_dist)
            
            if dist[neighbor] > times:
                dist[neighbor] = times
                heapq.heappush(pq, (dist[neighbor] + euclid, neighbor))
            
            if neighbor == target:
                return (dist[neighbor] * 60)
    
    return -1