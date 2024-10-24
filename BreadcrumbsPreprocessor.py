from collections import defaultdict
import sqlite3
import os
import pandas as pd
from scipy.spatial import KDTree
import csv
# from geopy import distance
from tqdm import tqdm
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from datetime import datetime

curDir = os.path.dirname(os.path.realpath(__file__))
dbCursor = sqlite3.connect(os.path.join(curDir,"../breadcrumbs_db.db"))

# Utility functions

# Save a list of dictionaries to a CSV
def saveCsv(fieldnames, rows, filename, filepath=curDir, dictWriterKwargs=None):
    if dictWriterKwargs == None:
        dictWriterKwargs = {}
    with open(os.path.join(filepath, filename), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,  **dictWriterKwargs)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

# Convert latitude / longitude to XYZ coordinates in 
def latLonToCartesian(lat, lon):
    R = 6371 #km, radius of earth
    return (R*np.cos(lat)*np.sin(lon), R*np.cos(lat)*np.cos(lon), R*np.sin(lat))

# TODO: Computational efficiency improvements ideas:
# Improved nearest-neighbor data structure: 
# https://stackoverflow.com/questions/1901139/closest-point-to-a-given-point
#
#
# Computational accuracy improvement ideas:
# Hysterisis for POI
#
# Use geodesic distance instead of lat/lon:
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors


def buildKDTree(poiData):
    # Project to 3d euclidean space and then use KDTree, as explained here:
    # https://cs.stackexchange.com/questions/48128/find-k-nearest-neighbors-on-a-sphere
    poiLatLons = list(pois[["latitude", "longitude"]].itertuples(index=False, name=None))
    poiCartesianPts = [latLonToCartesian(lat, lon) for (lat, lon) in poiLatLons]
    return KDTree(poiCartesianPts)

def parsePings(tree, pings):
    # ts = []
    ts_120m = []
    max_dist = .120 #km
    max_speed = 50 # guessing this is km/h
    for _, ping in tqdm(pings.iterrows(), total=len(pings)):
        pingCartesian = latLonToCartesian(ping["latitude"], ping["longitude"])
        dist, nearestPoiIndex = tree.query(pingCartesian)
        event = {
            "timestamp": ping["timestamp"], 
            "poi": nearestPoiIndex,
            "ping": ping,
            "user_id": ping["user_id"]
        }
        # ts.append(event)
        if dist < max_dist and ping["speed"] < max_speed: # dist in km. very slightly incorrect, since this is the euclidean dist
            ts_120m.append(event)
    return ts_120m

def chunkedPings(series):
    # nodePositions = {poiId: poiLatLons[poiId][::-1] for poiId in allPoisVisited}
    timestep = 4*60*60 # 4 hours
    # lookback_window = 1*60*60 # 1 hour #TODO
    t = min([t["timestamp"] for t in series]) + timestep
    i = 0 
    pingsByChunk = {}
    while i < len(series):
        # poiVisitsThisChunk["time"] =  datetime.fromtimestamp(t)
        pingsByChunk[t] = []
        while i < len(series) and t > series[i]["timestamp"]:
            pingsByChunk[t].append(series[i])
            i += 1
        t += timestep
    return pingsByChunk

# Generate the user transitions for each chunk, and from the end of one chunk to
# the beginning of the next chunk
def chunkTransitions(pingsByChunk):
    lastPingsInChunk = {}
    transitionsByChunk = {}
    for t, chunk in pingsByChunk.items():
        # pingsByUser = {user: pings[-1:] for user, pings in pingsByUser.items()} # reset, but keep last item
        pingsByUser = defaultdict(list)
        for ping in chunk:
            user = ping["user_id"]
            pingsByUser[user].append(ping)
        transitions = []
        for user, pings in pingsByUser.items():
            prependedPings = ([lastPingsInChunk[user]] if user in lastPingsInChunk else []) + pings
            if not prependedPings:
                import pdb
                pdb.set_trace()
            pairs = zip(prependedPings[:-1], prependedPings[1:])
            for lastPing, thisPing in pairs:
                if lastPing["poi"] != thisPing["poi"]:
                    edge = (int(lastPing["poi"]), int(thisPing["poi"]))
                    transitions.append(edge)
        transitionsByChunk[t] = transitions
        lastPingsInChunk = {user: pings[-1] for user, pings in pingsByUser.items()}
    return transitionsByChunk

def poiVisitsByChunk(pingsByChunk):
    allPois = set(ping["poi"] for t, chunk in pingsByChunk.items() for ping in chunk)
    visitsByChunk = {}
    for t, chunk in pingsByChunk.items():
        pingsByUser = defaultdict(list)
        for ping in chunk:
            user = ping["user_id"]
            pingsByUser[user].append(ping)
        visitsByUser = {user: set(ping["poi"] for ping in pings) for user, pings in pingsByUser.items()}
        visitsByChunk[t] = {poi: sum((1 if poi in visits else 0) for visits in visitsByUser.values()) for poi in allPois}
    return visitsByChunk

def buildDiGraph(nodes, edges):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def saveGraph(G, nodePositions, filename):
    # TODO: There are a few outliers in the data; would be nice to limit this to only pings near the epicenter
    nx.draw_networkx(G, nodePositions, node_size=10, with_labels=False, arrows=False) 
    plt.savefig(filename, dpi=600, transparent=True)
    plt.clf()


# TODO: Analyze data cleanliness: pings per hour
# TODO: Take David's learned data and map it

if __name__ == "__main__":
    # Read the data, build the KD Tree, generate the raw timeseries data
    pings = pd.read_sql("SELECT timestamp, latitude, longitude, speed, user_id FROM location ORDER BY timestamp LIMIT 500000;", dbCursor, dtype_backend='numpy_nullable')
    pois = pd.read_sql("SELECT id, latitude, longitude FROM point_of_interest;", dbCursor, dtype_backend='numpy_nullable')
    kdtree = buildKDTree(pois)
    ts_120m = parsePings(kdtree, pings)
    saveCsv(["timestamp", "poi", "user_id"], ts_120m, "allTimeseries.csv", dictWriterKwargs={"extrasaction":'ignore'})
    pingsByChunk = chunkedPings(ts_120m)
    allPoisVisited = set(ping["poi"] for t, chunk in pingsByChunk.items() for ping in chunk)
    transitionsByChunk = chunkTransitions(pingsByChunk)
    visitsByChunk = poiVisitsByChunk(pingsByChunk)
    saveCsv(["time", *allPoisVisited], [{**row, **{"time":datetime.fromtimestamp(t)}} for t, row in visitsByChunk.items()], "timeseriesByPOI.csv")
    poiLatLons = list(pois[["latitude", "longitude"]].itertuples(index=False, name=None))
    nodePositions = {poiId: poiLatLons[poiId][::-1] for poiId in allPoisVisited}
    G = buildDiGraph(allPoisVisited, list(transitionsByChunk.values())[0])
    saveGraph(G, nodePositions, "chunk1.png")