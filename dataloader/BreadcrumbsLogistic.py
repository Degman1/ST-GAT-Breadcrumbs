import json
import pkg_resources
pkg_resources.require("networkx>=3.4")
from collections import defaultdict
import sqlite3
import os
import pandas as pd
import csv
from geopy import distance
from tqdm import tqdm
import numpy as np
import networkx as nx # requires 3.4, only available in python 3.10+
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from multiprocessing import Pool

curDir = os.path.dirname(os.path.realpath(__file__))
dbCursor = sqlite3.connect(os.path.join(curDir,"../breadcrumbs_db.db"))

def loadData():
    pois = pd.read_sql("SELECT id, latitude, longitude FROM point_of_interest;", dbCursor, dtype_backend='numpy_nullable')
    debugMode = False
    # Read the data, build the KD Tree, generate the raw timeseries data
    print("Loading data from SQLite")
    pings = pd.read_sql(f"SELECT timestamp, latitude, longitude, speed, user_id FROM location ORDER BY timestamp {"LIMIT 500000" if debugMode else ""};", dbCursor, dtype_backend='numpy_nullable')
    pingsByUser = defaultdict(list)
    for ping in pings.to_dict('records'):
        pingsByUser[ping["user_id"]].append(ping)
    return pingsByUser, pois

def chunkedPings(series):
    h = 1
    timestep = timedelta(hours=h)
    # lookback_window = 1*60*60 # 1 hour #TODO
    minTimestamp = min([t["timestamp"] for t in series])
    t = datetime.fromtimestamp(minTimestamp)
    # Round time to nearest hour mark
    t = t.replace(second=0, microsecond=0, minute=0, hour=h*(t.hour//h))
    t += timestep
    i = 0 
    pingsByChunk = {}
    while i < len(series):
        pingsByChunk[t] = []
        while i < len(series) and t > datetime.fromtimestamp(series[i]["timestamp"]):
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

def computeAverageChunkDistancesThread(threadArgs):
    m = .120 # "cutoff" value; Should be that f(m) = 1/2 
    s = 3/m # affects "slope" of the curve; specific value chosen so that f(2/3*m) = .9 (approx)
    (poi_pos, userChunks) = threadArgs
    ret = []
    for dt, chunk in userChunks.items():
        t = dt.strftime("%Y-%m-%dT-%H-%M")
        avg = 0
        for ping in chunk:
            d = distance.great_circle(poi_pos, (ping["latitude"], ping["longitude"])).km
            if d < 5*m:
                logistic = 1/(1+np.exp(-1*s*(m*m/d-d)))
                avg += logistic / len(chunk)
                # avg += (d/(-5*m) + 1 )/len(chunk)
        ret.append((t, avg))
    return ret


def getAverageLogisticPopulationByChunk(pingsByUserByChunk, poiRecords):
    # m = .120 # "cutoff" value; Should be that f(m) = 1/2 
    # s = 3/m # affects "slope" of the curve; specific value chosen so that f(2/3*m) = .9 (approx)
    timeseriesByLogistic = {}
    numPings = sum([len(chunk) for ucs in pingsByUserByChunk.values() for chunk in ucs.values()])
    total = len(poiRecords)
    with Pool(24) as p:
        with tqdm(total=total) as pbar:
            for poi in poiRecords:
                poi_id = poi["id"]
                poi_pos = (poi["latitude"], poi["longitude"])
                    # for userChunks in pingsByUserByChunk.values():
                avgsByChunk = p.map(
                    computeAverageChunkDistancesThread,
                    [(poi_pos, userChunks) for userChunks in pingsByUserByChunk.values()]
                )
                for avgs in avgsByChunk:
                    for (t, avg) in avgs:
                        if t not in timeseriesByLogistic:
                            timeseriesByLogistic[t] = {}
                        if poi_id not in timeseriesByLogistic[t]:
                            timeseriesByLogistic[t][poi_id] = 0
                        timeseriesByLogistic[t][poi_id] += avg
                pbar.update(1)
    return timeseriesByLogistic



if __name__ == "__main__":
    pingsByUser, pois = loadData()
    poiRecords = pois.to_dict('records')
    pingsByUserByChunk = {user: chunkedPings(userPings) for user, userPings in pingsByUser.items()}
    avgLogisticPopulationByTimeByPoi = getAverageLogisticPopulationByChunk(pingsByUserByChunk, poiRecords)
    with open("blahblah.csv", 'w', newline='') as f:
        headers = ["timestamp"] + [poi["id"] for poi in poiRecords]
        print(headers)
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for t, avgs in avgLogisticPopulationByTimeByPoi.items():
            row = {"timestamp": t, **avgs}
            writer.writerow(row)

    with open("avgLogisticObj.json", 'w') as f:
        json.dump(avgLogisticPopulationByTimeByPoi, f, indent=4)


# Ideas:
# - Remove POIs that are not visited by many people or visited very often, like the trip to france
# - Look for POIs that are too close together and combine them 

# Assignment:
# - Graph laplacian
# - biggest eigenvalues
# - Create transition matrix
# - Create different transition matrices based on morning / afternoon / evening
# - Start writing up information on Overleaf