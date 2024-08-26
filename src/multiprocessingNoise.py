from multiprocessing import Pool

from smoothingWithCurvatureSigns import performSmoothing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from datetime import datetime

from dtaidistance import dtw_ndim

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

match_id = 1384039

file_path_skillcorner = 'data/matches/Feb/1384039_tracking.csv'
lineup_path = 'data/matches/Feb/1384039_lineup.csv'
partialframes = read_csv_partially(file_path_skillcorner, 0, 1500000)

lineup = read_csv_partially(lineup_path, 0, 33)

def getTrajectories(teamName, lineup, partialframes):
    playerIDs = set()
    teamName = "FC Basel"

    for player in list(filter(lambda player: player[1] == teamName, lineup[1:])):
        playerIDs.add(str(player[2]))

    targetTrajs = {}
    for ID in playerIDs:
        targetTrajs[ID] = []

    for index, playerFrame in enumerate(partialframes):

        if(playerFrame[4] not in playerIDs or playerFrame[8] == 'True'):
            continue

        # defining datapoints d_i = [x_Coordinate, y_Coordinate, speed, time]
        if targetTrajs[playerFrame[4]] == [] or int(targetTrajs[playerFrame[4]][-1][-1][-1]) != (int(playerFrame[2]) - 1):
            targetTrajs[playerFrame[4]].append([])
            
        targetTrajs[playerFrame[4]][-1].append([float(playerFrame[5]), float(playerFrame[6]), 0, datetime.fromtimestamp(int(playerFrame[3])/1000.0), int(playerFrame[2])])
    return targetTrajs

targetTrajs = getTrajectories("FC Basel", lineup, partialframes)


def getDTWDistance(DF1, DF2):
    series1 = np.array(DF1[['x', 'y']], dtype=np.double)
    series2 = np.array(DF2[['x', 'y']], dtype=np.double)
    d = dtw_ndim.distance(series1, series2)
    return d

def noiseHelper(p1, p, p2, lateralNoise, longitudinalNoise):
    longVec = np.array([p2['x'] - p1['x'], p2['y'] - p1['y']])
    latVec = np.array([-longVec[1], longVec[0]])
    new_p = np.array([p['x'], p['y']])

    alpha = 0
    beta = 0
    
    if(lateralNoise[0] == 'normal'):
        alpha = np.random.normal(lateralNoise[1], lateralNoise[2])
    
    if(longitudinalNoise[0] == 'normal'):
        beta = np.random.normal(longitudinalNoise[1], longitudinalNoise[2])

    new_p = new_p + alpha*latVec + beta*longVec

    return (new_p[0], new_p[1])

def reformatDF(df):
    df.columns = ['x', 'y', 'speed', 'time', 'frame_id']
    return df

def applyNoise(trajectory, lateralNoiseType, longitudinalNoiseType):
    trajectoryCopy = trajectory.copy()
    for i, row in trajectoryCopy.iterrows():
        if i == 0 or i == len(trajectoryCopy) - 1:
            continue
        trajectoryCopy.loc[i, 'x'], trajectoryCopy.loc[i, 'y'] = noiseHelper(trajectory.loc[i-1], trajectory.loc[i], trajectory.loc[i+1], lateralNoiseType, longitudinalNoiseType)
    
    return trajectoryCopy


allTrajs = targetTrajs.items()
dtwDistancesToOriginal = np.zeros(8, float)
dtwDistancesToSmoothened = np.zeros(8, float)
numTrajs = 0
its = [1, 2, 4, 10, 50, 100, 200, 400]

def perTrajectory(trajectory):
    dtwOriginal = np.zeros(8, float)
    print(len(trajectory), "Start of Smoothing")
    dtwSmooth = np.zeros(8, float)
    trajectory = pd.DataFrame(trajectory)
    trajectory.columns = ['x', 'y', 'speed', 'time', 'frame_id']
    noisyTrajectory = applyNoise(trajectory.copy(), ['normal', 0, 0.6], ['normal', 0, 1.2])

    smoothTrajs = [
        performSmoothing(trajectory[['x', 'y', 'speed', 'time', 'frame_id']].values.tolist(), 1),
        performSmoothing(trajectory[['x', 'y', 'speed', 'time', 'frame_id']].values.tolist(), 2),
        performSmoothing(trajectory[['x', 'y', 'speed', 'time', 'frame_id']].values.tolist(), 4),
        performSmoothing(trajectory[['x', 'y', 'speed', 'time', 'frame_id']].values.tolist(), 10),
        performSmoothing(trajectory[['x', 'y', 'speed', 'time', 'frame_id']].values.tolist(), 50),
        performSmoothing(trajectory[['x', 'y', 'speed', 'time', 'frame_id']].values.tolist(), 100),
        performSmoothing(trajectory[['x', 'y', 'speed', 'time', 'frame_id']].values.tolist(), 200),
        performSmoothing(trajectory[['x', 'y', 'speed', 'time', 'frame_id']].values.tolist(), 400),
    ]
    noisyTrajs = [
        performSmoothing(noisyTrajectory[['x', 'y', 'speed', 'time', 'frame_id']].values.tolist(), 1),
        performSmoothing(noisyTrajectory[['x', 'y', 'speed', 'time', 'frame_id']].values.tolist(), 2),
        performSmoothing(noisyTrajectory[['x', 'y', 'speed', 'time', 'frame_id']].values.tolist(), 4),
        performSmoothing(noisyTrajectory[['x', 'y', 'speed', 'time', 'frame_id']].values.tolist(), 10),
        performSmoothing(noisyTrajectory[['x', 'y', 'speed', 'time', 'frame_id']].values.tolist(), 50),
        performSmoothing(noisyTrajectory[['x', 'y', 'speed', 'time', 'frame_id']].values.tolist(), 100),
        performSmoothing(noisyTrajectory[['x', 'y', 'speed', 'time', 'frame_id']].values.tolist(), 200),
        performSmoothing(noisyTrajectory[['x', 'y', 'speed', 'time', 'frame_id']].values.tolist(), 400),
    ]
    print(len(trajectory), "Computing DTW")
    for i, s in enumerate(smoothTrajs):
        dtwOriginal[i] = getDTWDistance(trajectory, reformatDF(pd.DataFrame(noisyTrajs[i])))
        dtwSmooth[i] = getDTWDistance(reformatDF(pd.DataFrame(s)), reformatDF(pd.DataFrame(noisyTrajs[i])))

    return dtwOriginal, dtwSmooth


if __name__ == '__main__':
    filteredTrajs = {}
    for player, playerTrajs in allTrajs:
        for traj in playerTrajs:
            if(len(traj) > 300):
                if player not in filteredTrajs:
                    filteredTrajs[player] = []
                filteredTrajs[player].append(traj)

    with Pool(12) as p:
        for (player, playerTrajs) in filteredTrajs.items():
            if(len(playerTrajs) < 10):
                continue
            numTrajs += len(playerTrajs)
            for origdist, smoothdist in p.map(perTrajectory, playerTrajs):
                dtwDistancesToOriginal += origdist
                dtwDistancesToSmoothened += smoothdist
            if(numTrajs > 20):
                break

        plt.plot(its, dtwDistancesToOriginal/numTrajs, label='Original to smoothened noisy')
        plt.plot(its, dtwDistancesToSmoothened/numTrajs, label='Smoothened original to smoothened noisy')
        plt.xlabel('Iterations') 
        plt.ylabel('DTW Distance') 
        plt.legend()
        plt.show()
            
