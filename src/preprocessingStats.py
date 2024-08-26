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

# Format: (last_name, skillcorner_id, statsperform_id)
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

def getTrajectories(teamID, lineup, partialframes):
    playerIDs = set()

    for player in list(filter(lambda player: int(player[1]) == 1, lineup)):
        playerIDs.add(str(player[9]))
    
    print(playerIDs)

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
