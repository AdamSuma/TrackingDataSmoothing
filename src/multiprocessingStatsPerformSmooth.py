import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import isnan
from smoothingWithCurvatureSigns import performSmoothing, euclidianDistance
import time
from multiprocessing import Pool
from preprocessingStats import read_partially, read_csv_partially, getTrajectories, getSkillcornerTrajectories
import pickle


players = [
    ('Barry', 584967, 1397296),         # Thierno Barry
    ('Avdullahu', 659794, 1447729),     # Leon Avdullahu
    ('Sigua', 689056, 1442029),         # Gabriel Sigua
    ('Beney', 666916, 1448139),         # Roméo Beney
    ('Dräger', 19420, 795106),          # Mohamed Dräger
    ('Palma Veiga', 127332, 1341620),   # Renato Palma Veiga
    # ('Salvi', 7679, 374271),            # Mirko Salvi
    ('van Breemen', 162993, 1385368),   # Finn van Breemen
    ('Frei', 3233, 402048),             # Fabian Frei
    ('Schmid', 59893, 1050314),         # Dominik Robin Schmid
    ('Kade', 102835, 1189563),          # Anton Kade
    ('Jovanović', 14595, 976105),       # Djordje Jovanović
    ('Xhaka', 10218, 552970),           # Taulant Ragip Xhaka
    ('Dubasin', 258924, 1256449),       # Jonathan J. Dubasin
    ('Rüegg', 16280, 953869),           # Kevin Ruegg
]
# add for loop that starts here and smootens every trajectory in the traj list

small_its = [
    0, 
    2,
    5, 
    10,
    25,
    30,
    40,
    50,
]
big_its = [
    # 50,
    # 100, 
    # 150,
    # 200
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

def perIT(input):
    it = input[0]
    trajs = input[1]
    targetedPlayerID = input[2]
    print(it)
    playerSmoothDataFrames = pd.DataFrame()
    start_time = time.time()
    trajectories = []
    for traj in trajs:
        if(len(traj) < 2):
            continue
        trajectories.append((targetedPlayerID, traj, it))
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

    file_path_stats_perform = 'data/matches/vs FC Lugano/2023-12-06_StatsPerform_FC Lugano - FC fifa format Basel.txt'
    lineup_path_stats_perform = 'data/matches/vs FC Lugano/2023-12-06_StatsPerform_FC Lugano - FC Basel_Match-Info.csv'
    partialframes = read_partially(file_path_stats_perform, 0, 1500000)

    lineup = read_csv_partially(lineup_path_stats_perform, 0, 33)
    targetedPlayerID = 1050314

    targetTrajs = getTrajectories(1, lineup, partialframes)

    #  -------------------------------------------------------

    file_path_skillcorner = 'data/matches/vs FC Lugano/Preprocessed Skillcorner data/1296476_tracking.csv'
    lineup_path = 'data/matches/vs FC Lugano/Preprocessed Skillcorner data/1296476_lineup.csv'
    partialframes = read_csv_partially(file_path_skillcorner, 0, 1500000)

    lineup = read_csv_partially(lineup_path, 0, 33)
    targetPositionsSkillcorner = []
    skillcornerTrajs = getSkillcornerTrajectories("FC Basel", lineup, partialframes)
    targetSkillcornerID = 59893
    targetSkillcornerTrajs = skillcornerTrajs[str(targetSkillcornerID)]


    for name, skillcornerID, statsID in players:
        print(f'Preprocessing Player: {name}')
        newTrajs = []
        for i, traj in enumerate(skillcornerTrajs[str(skillcornerID)]):
            newTraj = []
            start = traj[0][-2]
            stop = traj[-1][-2]

            for statsTraj in targetTrajs[str(statsID)]:
                if statsTraj[-1][-2] < start or statsTraj[0][-2] > stop or int(statsTraj[0][-1]) != int(traj[0][-3]):
                    continue
                else:
                    for frame in statsTraj:
                        if frame[-2] < start:
                            continue
                        if frame[-2] > stop:
                            break
                        newTraj.append(frame)
            newTrajs.append(newTraj)
        targetTrajs[str(statsID)] = newTrajs

    totalMetrics = pd.DataFrame()

    for name, skillCornerID, statsPerformID in players:
        # if(name != 'Dräger'):
        #     continue
        start_time_player = time.time()
        print(f'Computing metrics for Player: {name}')
        targetedPlayerID = statsPerformID

        trajs = targetTrajs[str(targetedPlayerID)]

        smoothDataFrames = []
        with Pool(8) as p:
            for playerSmoothDataFrames in p.map(perIT, [(it, trajs, statsPerformID) for it in small_its]):
                smoothDataFrames.append(playerSmoothDataFrames)

        with Pool(14) as p:
            for it in big_its:
                print(it)
                start_time = time.time()
                playerSmoothDataFrames = pd.DataFrame()
                start_time = time.time()
                trajectories = []
                for traj in trajs:
                    if(len(traj) < 2):
                        continue
                    trajectories.append((targetedPlayerID, traj, it))
                
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

        smoothDataFrames[0] = originaltargetedPlayerDF

        print(extractMetrics(originaltargetedPlayerDF))

        metrics = []
        for df in smoothDataFrames:
            metrics.append(extractMetrics(df))
            
        maxSpeedList = list(map(lambda m: m['maxSpeed'], metrics))

        metrics = pd.DataFrame(metrics)
        metrics['iterations'] = small_its + big_its
        metrics['name'] = name
        metrics = metrics[['name', 'iterations', 'maxSpeed', 'maximumSpeedSustained', 'sprintCount', 'distanceCovered']]

        totalMetrics = pd.concat([totalMetrics, metrics], axis=0)  
        stop_time_player = time.time()
        print(f'Player {name} took {stop_time_player - start_time_player} seconds')   
        print(metrics)

    with open('./src/dataframes/withCurvatureSign/totalMetricsStatsPerformSmallerIts.pkl', 'wb') as file:
        pickle.dump(totalMetrics, file)