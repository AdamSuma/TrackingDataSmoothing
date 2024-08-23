import numpy as np
import math
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def getCircle(points):
    assert(len(points) == 3)
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]

    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    if math.isclose(D, 0):
        print('Points are collinear')
        return None

    Ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
    Uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D

    cross_product = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

    r = math.sqrt((x1 - Ux)**2 + (y1 - Uy)**2)

    # If movement clockwise return sign is negative, positive otherwise
    if cross_product < 0:
        r = -r

    return ((Ux, Uy), r)

def getMovementSign(c, p1, p2): # takes as input the center of the circle and two points on the circle, returns -1 if the movement is clockwise and 1 if it is counterclockwise
    return -1 if (p1[0]-c[0])*(p2[1]-c[1]) - (p1[1]-c[1])*(p2[0]-c[0]) < 0 else 1

def getCircleIntersections(c0, r0, c1, r1):
    x0, y0 = c0
    x1, y1 = c1
    d = math.sqrt((x1-x0)**2 + (y1-y0)**2)
    
    # case tangent circles
    if(math.isclose(d, r0+r1, rel_tol=1e-10)):
        return (-1, -1)

    # non intersecting
    if d > r0 + r1 :
        print('non intersecting')
        return None
    # One circle within other
    if d < abs(r0-r1):
        print('One circle within other')
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        print(' coincident circles')
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d
        
        return ((x3, y3), (x4, y4))
    
def euclidianDistance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def closestPoint(p1, p2, p3, p4):
    if(euclidianDistance(p1, p3) + euclidianDistance(p1, p4) < euclidianDistance(p2, p3) + euclidianDistance(p2, p4)):
        return p1
    else:
        return p2
    
def areCollinear(p1, p2, p3):
    x1, y1 = p2[0] - p1[0], p2[1] - p1[1]
    x2, y2 = p3[0] - p1[0], p3[1] - p1[1]
    return abs(x1 * y2 - x2 * y1) < 1e-12

def midPoint(p1, p2):
    return ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)

def getAngle(p1, p2, p3):
    a = np.array([p1[0], p1[1]])
    b = np.array([p2[0], p2[1]])
    c = np.array([p3[0], p3[1]])
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    # print(f'Cosine Angle {cosine_angle}')
    # print(f'ba: {ba}, bc: {bc}')
    assert(angle <= 2*math.pi and angle >= 0)
    return angle

def computeNewPoints(datapoints):
    new_points = []
    trajectory = list(map(lambda x: x[:2], datapoints)) 
    n = len(trajectory)

    for i in range(1, n-2):
        if(i == 1):
            p1 = trajectory[i]
            p2 = trajectory[i+1]
            # We assume that between frames the time interval is kept constant. If this is not the case move t12 and t12delta outside of the if statement
            t12 = (datapoints[i+1][3] - datapoints[i][3]).total_seconds()
            t12delta = timedelta(seconds=t12/2)
            col1 = areCollinear(trajectory[i-1], p1, p2)            
        else:
            p1 = p2
            c1 = c2
            r1 = r2
            col1 = col2
    
        p2 = trajectory[i+1]
        M = midPoint(p1, p2)
        
        # handling collinear points
        col2 = areCollinear(trajectory[i+2], p1, p2)

        if(not(col1 or col2)):
            if(i == 1):
                c1, r1 = getCircle(trajectory[i-1: i+2])
            c2, r2 = getCircle(trajectory[i: i+3])
            r = 2*r1*r2/(r1+r2)

        elif(col1 and col2):
            new_point = M
            new_points.append([new_point[0], new_point[1], euclidianDistance(p1, p2), datapoints[i][3] + t12delta])
            c2 = (0, 0)
            r2 = float('inf')
            continue

        elif(col1):
            c2, r2 = getCircle(trajectory[i: i+3])
            c1 = (0, 0)
            r1 = float('inf')
            r = 2*r2

        elif(col2):
            if(i == 1):
                c1, r1 = getCircle(trajectory[i-1: i+2])
            c2 = (0, 0)
            r2 = float('inf')
            r = 2*r1

        # intersecting the averaging circles
        c_avg1, c_avg2 = getCircleIntersections(p1, abs(r), p2, abs(r))

        if(c_avg1 == -1 or c_avg2 == -1 or abs(r) > 10e6):
            new_point = M
            new_points.append([new_point[0], new_point[1], euclidianDistance(p1, p2), datapoints[i][3] + t12delta])
            continue

        cS1 = getMovementSign(c_avg1, p1, p2)
        cS2 = getMovementSign(c_avg2, p1, p2)
        if not((cS1 >= 0 and cS2 <= 0) or (cS1 <= 0 and cS2 >= 0)):
            assert(False)

        if(getMovementSign(c_avg1, p1, p2)*r >= 0):
            c_avg = c_avg1
        else:
            c_avg = c_avg2

        # computing new point
        d = abs(r)/euclidianDistance(M, c_avg)
        new_point_1 = np.array(c_avg) + (np.array(M)-np.array(c_avg))*d
        new_point_2 = np.array(c_avg) + (np.array(M)-np.array(c_avg))*(-d)

        # print(p1, c_avg, p2)
        # print(f'Center {c_avg}')
        # print(f'Radius {r}')

        new_point = closestPoint(new_point_1, new_point_2, p1, p2)
        new_point = [new_point[0], new_point[1], abs(r)*getAngle(p1, c_avg, p2), datapoints[i][3] + t12delta]

        new_points.append(new_point)

    return new_points

def performSmoothing(datapoints, iterations):
    
    # compute extension points at the beginning/end of the trajectory
    pFirst = [2*datapoints[0][0] - datapoints[1][0], 2*datapoints[0][1] - datapoints[1][1], 0, datapoints[0][3] - (datapoints[1][3] - datapoints[0][3])]
    pLast = [2*datapoints[-1][0] - datapoints[-2][0], 2*datapoints[-1][1] - datapoints[-2][1], 0, datapoints[-1][3] + (datapoints[-1][3] - datapoints[-2][3])]
    midFirst = midPoint(pFirst, datapoints[0])
    midLast = midPoint(datapoints[-1], pLast)

    # midFirst = [midFirst[0], midFirst[1], 0, datapoints[0][3] - (datapoints[1][3] - datapoints[0][3])/2]
    # midLast = [midLast[0], midLast[1], 0, datapoints[-1][3] + (datapoints[-1][3] - datapoints[-2][3])/2]

    midFirst = [midFirst[0], midFirst[1], 2*euclidianDistance(midFirst, pFirst), datapoints[0][3] - (datapoints[1][3] - datapoints[0][3])/2]
    midLast = [midLast[0], midLast[1], 2*euclidianDistance(midLast, pLast), datapoints[-1][3] + (datapoints[-1][3] - datapoints[-2][3])/2]
    pFirst[2] = 2*euclidianDistance(pFirst, midFirst)
    pLast[2] = 2*euclidianDistance(pLast, midLast)

    for _ in range(0, iterations):
        for i in range(0, 2):
            if(i % 2 != 0):
                og = datapoints.copy()
                updated_datapoints = computeNewPoints([midFirst] + new_points + [midLast])
                datapoints = [og[0]] + updated_datapoints + [og[-1]]
            else:
                new_points = computeNewPoints([pFirst] + datapoints + [pLast])
    return datapoints