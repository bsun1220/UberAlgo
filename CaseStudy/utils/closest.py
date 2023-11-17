from utils.constants import *
from math import sqrt

def slow_get_closest_node(coord, nodes):
    closest_node = None
    closest_dist = float('inf')
    
    coord_lon, coord_lat = coord['lon'], coord['lat']
    
    for node in nodes:
        node_lon, node_lat = nodes[node]['lon'], nodes[node]['lat']
        dist = (node_lon - coord_lon)**2 + (node_lat - coord_lat)**2
        if dist < closest_dist:
            closest_dist = dist
            closest_node = node
    return closest_node

def fast_get_closest_node(coord, nodes, threshold = 1, init = True):
    
    sorted_nodes = []
    if not init:
        nodes_df = pd.DataFrame(nodes).T.reset_index()
        sorted_nodes = []
        for d in nodes_df.itertuples():
            row = (d[2], d[3], d[1])
            sorted_nodes.append(row)
        sorted_nodes.sort()
        with open("data/sorted_nodes.pkl", "wb") as outfile:
            pickle.dump(sorted_nodes, outfile)
    else:
        sorted_nodes = pd.read_pickle('data/sorted_nodes.pkl')
    
    left, mid, right = 0, 0, len(sorted_nodes) - 1
    while left <= right:
        mid = (left + right)//2
        if sorted_nodes[mid][0] < coord['lon']:
            left = mid + 1
        elif sorted_nodes[mid][0] > coord['lon']:
            right = mid - 1
        else:
            break
    
    closest_node = sorted_nodes[mid][2]
    l1 = (sorted_nodes[mid][0] - coord['lon'])
    l2 = (sorted_nodes[mid][1] - coord['lat'])
    closest_dist = sqrt(l1 * l1 + l2 * l2)
    
    left, mid_low, right = 0, 0, mid - 1
    target_low = sorted_nodes[mid][0] - threshold * closest_dist
    while left <= right:
        mid_low = (left + right)//2
        if sorted_nodes[mid_low][0] < target_low:
            left = mid_low + 1
        elif sorted_nodes[mid_low][0] > target_low:
            right = mid_low - 1
        else:
            break
    
    left, mid_high, right = mid + 1, len(sorted_nodes) - 1, len(sorted_nodes) - 1
    target_high = sorted_nodes[mid][0] + threshold * closest_dist
    while left <= right:
        mid_high = (left + right)//2
        if sorted_nodes[mid_high][0] < target_high:
            left = mid_high + 1
        elif sorted_nodes[mid_low][0] > target_high:
            right = mid_high - 1
        else:
            break
    
    lon, lat = coord['lon'], coord['lat']
    for i in range(mid_low, mid_high + 1):
        row = sorted_nodes[i]
        l1 = (row[0] - lon)
        l2 = (row[1] - lat)
        dist = sqrt(l1*l1 + l2*l2)
        if dist <= closest_dist:
            closest_dist = dist
            closest_node = row[2]
    
    return closest_node