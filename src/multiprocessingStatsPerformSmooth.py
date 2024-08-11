import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from mplsoccer.pitch import Pitch
import json
import pandas as pd
import re
from math import isnan, sqrt
from datetime import datetime, timedelta
from smoothing import performSmoothing, euclidianDistance
import time

from multiprocessing import Pool

import csv 

def read_partially(file_path, start_line, end_line):
    data = []  # List to store extracted CSV rows

    with open(file_path, 'r', newline='') as file:
        for line_num, row in enumerate(file):
            if line_num >= start_line and line_num <= end_line:
                # Append the row to data
                data.append(row)
            elif line_num > end_line:
                break  # Stop reading once end_line is reached

    return data
import csv

def read_csv_partially(file_path, start_line, end_line):
    data = []  # List to store extracted CSV rows

    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        for line_num, row in enumerate(reader):
            if line_num >= start_line and line_num <= end_line:
                # Append the row to data
                data.append(row)
            elif line_num > end_line:
                break  # Stop reading once end_line is reached

    return data

file_path_skillcorner = 'data/matches/vs FC Lugano/2023-12-06_StatsPerform_FC Lugano - FC fifa format Basel.txt'
lineup_path = 'data/matches/vs FC Lugano/2023-12-06_StatsPerform_FC Lugano - FC Basel_Match-Info.csv'
partialframes = read_partially(file_path_skillcorner, 0, 1500000)

lineup = read_csv_partially(lineup_path, 0, 33)
# targetIDSkillcorner = 7679
targetedPlayerID = 1050314
targetPositionsSkillcorner = []

def getTrajectories(teamID, lineup, partialframes):
    playerIDs = set()

    for player in list(filter(lambda player: int(player[1]) == 1, lineup)):
        playerIDs.add(str(player[9]))
    
    # print(playerIDs)
    # return None
    targetTrajs = {}
    for ID in playerIDs:
        targetTrajs[ID] = []

    for index, frame in enumerate(partialframes):
        f = re.split(':', frame)
        frameVars = re.split(';|,', f[0]) # 0 - system time; 1 - milliseconds of current half; 2 - current half
        frameVars = list(map(lambda x: int(x), frameVars))
        # print(frameVars)
        # return 
        playerPositions = f[1].split(';')
        for playerString in playerPositions:
            playerFrame = playerString.split(',') # 0 - object type; 1 - playerID; 2 - shirt Number, 3 - x; 4 - y
            if(len(playerFrame)) < 5:
                continue
            if(playerFrame[1] not in playerIDs):
                continue
            if targetTrajs[playerFrame[1]] == [] or int(targetTrajs[playerFrame[1]][-1][-1][-2]) != (int(frameVars[1]) - 40):
                targetTrajs[playerFrame[1]].append([])
            targetTrajs[playerFrame[1]][-1].append([float(playerFrame[3]), float(playerFrame[4]), -1, datetime.fromtimestamp(int(frameVars[1])/1000.0), int(frameVars[0]), int(frameVars[1]), int(frameVars[2])])
    return targetTrajs

targetTrajs = getTrajectories(1, lineup, partialframes)

def getSkillcornerTrajectories(teamName, lineup, partialframes):
    playerIDs = set()

    for player in list(filter(lambda player: player[1] == teamName, lineup[1:])):
        playerIDs.add(str(player[2]))

    targetTrajs = {}
    for ID in playerIDs:
        targetTrajs[ID] = []

    for index, playerFrame in enumerate(partialframes):

        if(playerFrame[4] not in playerIDs or playerFrame[8] == 'True'):
            continue

        if targetTrajs[playerFrame[4]] == [] or int(targetTrajs[playerFrame[4]][-1][-1][-1]) != (int(playerFrame[2]) - 1):
            targetTrajs[playerFrame[4]].append([])
        
        targetTrajs[playerFrame[4]][-1].append([float(playerFrame[5]), float(playerFrame[6]), -1, datetime.fromtimestamp(int(playerFrame[3])/1000.0), int(playerFrame[1]), int(playerFrame[3]), int(playerFrame[2])])
    return targetTrajs

file_path_skillcorner = 'data/matches/vs FC Lugano/Preprocessed Skillcorner data/1296476_tracking.csv'
lineup_path = 'data/matches/vs FC Lugano/Preprocessed Skillcorner data/1296476_lineup.csv'
partialframes = read_csv_partially(file_path_skillcorner, 0, 1500000)

lineup = read_csv_partially(lineup_path, 0, 33)
# targetIDSkillcorner = 7679
targetPositionsSkillcorner = []

skillcornerTrajs = getSkillcornerTrajectories("FC Basel", lineup, partialframes)
targetSkillcornerID = 59893
targetSkillcornerTrajs = skillcornerTrajs[str(targetSkillcornerID)]
# print(len(skillcornerTrajs))
# print(targetSkillcornerTrajs[0])
newTrajs = []
currentStatsPerformTrajIDX = 0
# print(len(targetSkillcornerTrajs))

for i, traj in enumerate(targetSkillcornerTrajs):
    newTraj = []
    start = traj[0][-2]
    stop = traj[-1][-2]
    # print(start)
    # print(stop)
    # print(f'First point in Skillcorner Traj ----------- {traj[0]}')
    for statsTraj in targetTrajs[str(targetedPlayerID)]:
        if statsTraj[-1][-2] < start or statsTraj[0][-2] > stop or int(statsTraj[0][-1]) != int(traj[0][-3]):
            continue
        else:
            for frame in statsTraj:
                if frame[-2] < start:
                    continue
                if frame[-2] > stop:
                    break
                # print(f'Frame in StatsPerform --- {frame}')
                if(frame[1] < 0 or frame[0] < 0):
                    print(frame)
                newTraj.append(frame)
    # print(len(newTraj))
    # print(len(traj))
    # if(i >= 6):
    #     break
    newTrajs.append(newTraj)
targetTrajs[str(targetedPlayerID)] = newTrajs

# add for loop that starts here and smootens every trajectory in the traj list

small_its = [
    0, 
    2,
    5, 
    10,
]
big_its = [
    50,
    100, 
    150,
    200
]
def perTraj(args):
    player = args[0]
    traj = args[1]
    it = args[2]

    new_traj = performSmoothing(datapoints=traj, iterations=it)
    new_traj = pd.DataFrame(new_traj)
    new_traj[5] = str(player)
    traj = pd.DataFrame(traj)
    for idx, row in traj.iterrows():
        if(isnan(row[4])):
            print('nan')

    new_traj[4] = traj[4]
    new_traj[7] = traj[6]
    new_traj[6] = traj[5]
    return new_traj

def perIT(it):
    print(it)
    playerSmoothDataFrames = pd.DataFrame()
    start_time = time.time()
    trajs = targetTrajs[str(targetedPlayerID)]
    trajectories = []
    for traj in trajs:
        if(len(traj) < 2):
            continue
        trajectories.append((targetedPlayerID, traj, it))
    # print(str(len(trajectories)) + " len trajectories " + str(it))
    for smoothTraj in map(perTraj, trajectories):
        if smoothTraj is None:
            continue
        playerSmoothDataFrames = pd.concat([playerSmoothDataFrames, smoothTraj], axis=0, ignore_index=True)

    playerSmoothDataFrames[4] = playerSmoothDataFrames[4].astype(int)
    playerSmoothDataFrames.columns = ['x', 'y', 'arc_length', 'time', 'systemTime', 'object_id', 'timestamp', 'half']

    playerSmoothDataFrames['speed'] = playerSmoothDataFrames['arc_length']*25
    playerSmoothDataFrames.loc[playerSmoothDataFrames['arc_length'] == -1, 'speed'] = None

    playerSmoothDataFrames['systemTime'] = playerSmoothDataFrames['systemTime'].astype(int)
    playerSmoothDataFrames['timestamp'] = playerSmoothDataFrames['timestamp'].astype(int)
    playerSmoothDataFrames['half'] = playerSmoothDataFrames['half'].astype(int)
    playerSmoothDataFrames['x'] = playerSmoothDataFrames['x'].astype(float)
    playerSmoothDataFrames['y'] = playerSmoothDataFrames['y'].astype(float)

    stop_time = time.time()
    print(f'Iteration {it} took {stop_time - start_time} seconds')
    return playerSmoothDataFrames.copy()
    


def extractMetrics(playerDF):
    maxSpeed = 0
    distanceCovered = 0
    metrics = {
        'maxSpeed': playerDF['speed'].max(),
        'distanceCovered': 0,
        'maximumSpeedSustained': 0,
        'sprintCount': 0,
    }

    # get sustained speed
    idxmax = playerDF['speed'].idxmax()
    l = idxmax
    h = l
    while (h < len(playerDF) and l >= 0):
        if(abs(playerDF.loc[h, 'speed'] - metrics['maxSpeed']) < 0.28):
            h += 1
        elif(abs(playerDF.loc[l, 'speed'] - metrics['maxSpeed']) < 0.28):
            l -= 1
        else:
            break
    metrics['maximumSpeedSustained'] = playerDF.loc[h, 'timestamp'] - playerDF.loc[l, 'timestamp']
    startOfSprintIDX = -1
    # handle unsmoothed data
    if("arc_length" not in playerDF.columns):
        for i, row in playerDF.iterrows():
            if(i == 0):
                continue
            if(row['timestamp'] == playerDF.loc[i-1, 'timestamp'] + 40):
                distanceCovered += euclidianDistance([row['x'], row['y']], [playerDF.loc[i-1, 'x'], playerDF.loc[i-1, 'y']])

            if(row['speed'] > 6.94):
                if(startOfSprintIDX == -1):
                    startOfSprintIDX = i
            else:
                if(startOfSprintIDX != -1):
                    timeOfSprint = playerDF.loc[i-1, 'timestamp'] - playerDF.loc[startOfSprintIDX, 'timestamp']
                    if(timeOfSprint > 700):
                        metrics['sprintCount'] += 1
                    startOfSprintIDX = -1
            
    # handle smoothened data
    else:
        for i, row in playerDF.iterrows():
            if(i == 0):
                continue
            arclength = row['arc_length']
            prevArclength = playerDF.loc[i-1, 'arc_length']
            if(arclength == -1 and prevArclength == -1):
                continue
            if((arclength != -1 and prevArclength == -1) or (arclength == -1 and prevArclength != -1)):
                distanceCovered += euclidianDistance([row['x'], row['y']], [playerDF.loc[i-1, 'x'], playerDF.loc[i-1, 'y']])
            else:
                distanceCovered += arclength/2 + prevArclength/2

            if(row['speed'] > 6.94):
                if(startOfSprintIDX == -1):
                    startOfSprintIDX = i
            else:
                if(startOfSprintIDX != -1):
                    timeOfSprint = playerDF.loc[i-1, 'timestamp'] - playerDF.loc[startOfSprintIDX, 'timestamp']
                    if(timeOfSprint > 700):
                        metrics['sprintCount'] += 1
                    startOfSprintIDX = -1
            
                
    metrics['distanceCovered'] = distanceCovered   
    return metrics

if __name__ == '__main__':
    # smoothDataFrames = extractSmoothDataFrames(targetTrajs, targetedPlayerID, its)
    smoothDataFrames = []
    with Pool(8) as p:
        for playerSmoothDataFrames in p.map(perIT, small_its):
            smoothDataFrames.append(playerSmoothDataFrames)

    with Pool(14) as p:
        for it in big_its:
            print(it)
            playerSmoothDataFrames = pd.DataFrame()
            start_time = time.time()
            trajs = targetTrajs[str(targetedPlayerID)]
            trajectories = []
            for traj in trajs:
                if(len(traj) < 2):
                    continue
                trajectories.append((targetedPlayerID, traj, it))
            print(str(len(trajectories)) + " len trajectories " + str(it))
            for smoothTraj in p.map(perTraj, trajectories):
                if smoothTraj is None:
                    continue
                playerSmoothDataFrames = pd.concat([playerSmoothDataFrames, smoothTraj], axis=0, ignore_index=True)

            playerSmoothDataFrames[4] = playerSmoothDataFrames[4].astype(int)
            playerSmoothDataFrames.columns = ['x', 'y', 'arc_length', 'time', 'systemTime', 'object_id', 'timestamp', 'half']

            playerSmoothDataFrames['speed'] = playerSmoothDataFrames['arc_length']*25
            playerSmoothDataFrames.loc[playerSmoothDataFrames['arc_length'] == -1, 'speed'] = None

            playerSmoothDataFrames['systemTime'] = playerSmoothDataFrames['systemTime'].astype(int)
            playerSmoothDataFrames['timestamp'] = playerSmoothDataFrames['timestamp'].astype(int)
            playerSmoothDataFrames['half'] = playerSmoothDataFrames['half'].astype(int)
            playerSmoothDataFrames['x'] = playerSmoothDataFrames['x'].astype(float)
            playerSmoothDataFrames['y'] = playerSmoothDataFrames['y'].astype(float)

            stop_time = time.time()
            print(f'Iteration {it} took {stop_time - start_time} seconds')
            smoothDataFrames.append(playerSmoothDataFrames.copy())
            
    originalDF = smoothDataFrames[0].drop(columns=['arc_length'])
    smoothDataFrames.pop(0)

    originaltargetedPlayerDF = pd.DataFrame(originalDF[originalDF['object_id'] == str(targetedPlayerID)])
    # originaltargetedPlayerDF = pd.merge(originaltargetedPlayerDF, frames, how='right', on=['frame_id'])

    # originaltargetedPlayerDF['arc_length'] = np.nan 
    originaltargetedPlayerDF['x'] = originaltargetedPlayerDF['x'].astype(float)
    originaltargetedPlayerDF['y'] = originaltargetedPlayerDF['y'].astype(float)
    originaltargetedPlayerDF['timestamp'] = originaltargetedPlayerDF['timestamp'].astype(int)

    euclidean_distance1 = np.sqrt((originaltargetedPlayerDF['x'] - originaltargetedPlayerDF['x'].shift(1))**2 + (originaltargetedPlayerDF['y'] - originaltargetedPlayerDF['y'].shift(1))**2)
    euclidean_distance2 = np.sqrt((originaltargetedPlayerDF['x'] - originaltargetedPlayerDF['x'].shift(-1))**2 + (originaltargetedPlayerDF['y'] - originaltargetedPlayerDF['y'].shift(-1))**2)

    timediff1 = originaltargetedPlayerDF['timestamp'] - originaltargetedPlayerDF['timestamp'].shift(1)
    timediff2 = originaltargetedPlayerDF['timestamp'].shift(-1) - originaltargetedPlayerDF['timestamp']

    originaltargetedPlayerDF['speed'] = (euclidean_distance1+euclidean_distance2)/(timediff1+timediff2)*1000

    # Use the Series to fill the NaNs
    # originaltargetedPlayerDF['arc_length'] = originaltargetedPlayerDF['arc_length'].fillna((euclidean_distance1 + euclidean_distance2)/2)

    # originaltargetedPlayerDF['distance_covered'] = originaltargetedPlayerDF['arc_length'].cumsum() - euclidean_distance2/2
    # originaltargetedPlayerDF['distance_covered'] = originaltargetedPlayerDF['arc_length'].cumsum()

    originaltargetedPlayerDF['timediff'] = (originaltargetedPlayerDF['timestamp'] - originaltargetedPlayerDF['timestamp'].shift(1)) + (originaltargetedPlayerDF['timestamp'].shift(-1) - originaltargetedPlayerDF['timestamp'])
    # if timediff is larger than 200, put speed to NaN
    originaltargetedPlayerDF.loc[originaltargetedPlayerDF['timediff'] > 80, 'speed'] = np.nan

    print(extractMetrics(originaltargetedPlayerDF))

    metrics = []
    for df in smoothDataFrames:
        metrics.append(extractMetrics(df))
        
    maxSpeedList = list(map(lambda m: m['maxSpeed'], metrics))
    # plt.plot(its[1:], maxSpeedList, label='StatsPerform Data')
    # maxSpeedListSkillCorner = [8.705650944692918, 8.683963461906927, 8.64044410352124, 8.3178803099313, 7.982682713411052, 7.70187746256995, 7.460274849208323]
    # plt.plot(its[1:], maxSpeedListSkillCorner, label='SkillCorner Data')
    # plt.xlabel('Smoothing Iterations')  # Replace 'Index' with the actual x-axis label
    # plt.ylabel('Max Speed [m/s]')  # Replace 'Value' with the actual y-axis label  
    # plt.legend()
    # plt.show()
    metrics = pd.DataFrame(metrics)
    metrics['iterations'] = small_its[1:] + big_its
    metrics = metrics[['iterations', 'maxSpeed', 'maximumSpeedSustained', 'sprintCount', 'distanceCovered']]
    print(metrics)