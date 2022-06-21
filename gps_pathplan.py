# -*- coding: utf-8 -*-

"""
GPS based path planning
author: happylyrics
"""
# Libraries
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# Parameters
r = 5.0  #ArcRadius[m]
vehicle_velocity = 1.50 #RobotVelocity[m/s]

# Waypoint
waypoint0_lat = 33.841026
waypoint0_lon = 132.762288
waypoint1_lat = 33.841674
waypoint1_lon = 132.762299
waypoint2_lat = 33.841672
waypoint2_lon = 132.763082
waypoint3_lat = 33.842227
waypoint3_lon = 132.763082
waypoint4_lat = 33.842236
waypoint4_lon = 132.762310
waypoint5_lat = 33.841995
waypoint5_lon = 132.762302

# Variable init
count = 0
bak_count = 0
q = 0
mod = 0
bak_mod = 0
offset_x = 0
offset_y = 0
waypoint_x = []
waypoint_y = []
waypoint_xy = []
waypoint_0 = [0,0]
arc_count = 0
vector_12 = []
waypoint_ref = []
waypoint_lat = []
waypoint_lon = []
waypoint_list = []

# Add waypoint list
init_lat = waypoint0_lat
init_lon = waypoint0_lon
for i in range(1000):
    way_lat = 'waypoint'+str(i)+'_lat'
    way_lon = 'waypoint'+str(i)+'_lon'
    try:
        if globals()[way_lat]:
            waypoint_list.append([globals()[way_lat],globals()[way_lon]])
    except KeyError:
        break
print("waypointlist_len",len(waypoint_list))

# Ellipsoid
ELLIPSOID_GRS80 = 1 # GRS80
ELLIPSOID_WGS84 = 2 # WGS84

# Long axis radius and flatness by ellipsoid
GEODETIC_DATUM = {
    ELLIPSOID_GRS80: [
        6378137.0,         # [GRS80]長軸半径
        1 / 298.257222101, # [GRS80]扁平率
    ],
    ELLIPSOID_WGS84: [
        6378137.0,         # [WGS84]長軸半径
        1 / 298.257223563, # [WGS84]扁平率
    ],
}

# Maximum number of iterations
ITERATION_LIMIT = 1000


def vincenty_inverse(lat1, lon1, lat2, lon2, ellipsoid=None):

    # Return 0.0 if there is no difference
    if math.isclose(lat1, lat2) and math.isclose(lon1, lon2):
        return {
            'distance': 0.0,
            'azimuth1': 0.0,
            'azimuth2': 0.0,
        }

    # Obtain the necessary major axis radius (a) and flatness (ƒ) from the constants and calculate the minor axis radius (b) at the time of calculation
    # If the ellipsoid is not specified, the value of GRS80 is used.
    a, ƒ = GEODETIC_DATUM.get(ellipsoid, GEODETIC_DATUM.get(ELLIPSOID_GRS80))
    b = (1 - ƒ) * a

    φ1 = math.radians(lat1)
    φ2 = math.radians(lat2)
    λ1 = math.radians(lon1)
    λ2 = math.radians(lon2)

    # Modified latitude (latitude on the auxiliary sphere)
    U1 = math.atan((1 - ƒ) * math.tan(φ1))
    U2 = math.atan((1 - ƒ) * math.tan(φ2))

    sinU1 = math.sin(U1)
    sinU2 = math.sin(U2)
    cosU1 = math.cos(U1)
    cosU2 = math.cos(U2)

    # Longitude difference between two points
    L = λ2 - λ1

    # Initialize lambda with L
    λ = L

    # Iterate the following calculations until λ converges
    # Set an upper limit on the number of iterations, as convergence may not occur at some locations.
    for i in range(ITERATION_LIMIT):
        sinλ = math.sin(λ)
        cosλ = math.cos(λ)
        sinσ = math.sqrt((cosU2 * sinλ) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosλ) ** 2)
        cosσ = sinU1 * sinU2 + cosU1 * cosU2 * cosλ
        σ = math.atan2(sinσ, cosσ)
        sinα = cosU1 * cosU2 * sinλ / sinσ
        cos2α = 1 - sinα ** 2
        cos2σm = cosσ - 2 * sinU1 * sinU2 / cos2α
        C = ƒ / 16 * cos2α * (4 + ƒ * (4 - 3 * cos2α))
        λʹ = λ
        λ = L + (1 - C) * ƒ * sinα * (σ + C * sinσ * (cos2σm + C * cosσ * (-1 + 2 * cos2σm ** 2)))

        # If the deviation is less than .0000000000000001, break
        if abs(λ - λʹ) <= 1e-12:
            break
    else:
        # Returns None if the calculation does not converge
        return None

    # Once λ converges to the desired accuracy, the following calculation is performed
    u2 = cos2α * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    Δσ = B * sinσ * (cos2σm + B / 4 * (cosσ * (-1 + 2 * cos2σm ** 2) - B / 6 * cos2σm * (-3 + 4 * sinσ ** 2) * (-3 + 4 * cos2σm ** 2)))

    # Distance on the ellipsoid between two points
    s = b * A * (σ - Δσ)

    # Azimuth angle at each point
    α1 = math.atan2(cosU2 * sinλ, cosU1 * sinU2 - sinU1 * cosU2 * cosλ)
    α2 = math.atan2(cosU1 * sinλ, -sinU1 * cosU2 + cosU1 * sinU2 * cosλ) + math.pi

    if α1 < 0:
        α1 = α1 + math.pi * 2

    return {
        'distance': s,           # Distance
        'azimuth1': math.degrees(α1), # Azimuth (start point to end point)
        'azimuth2': math.degrees(α2), # Azimuth (end point to start point)
    }

def vincenty_direct(lat, lon, azimuth, distance, ellipsoid=None):

    # Obtain the necessary major axis radius (a) and flatness (ƒ) from the constants and calculate the minor axis radius (b) at the time of calculation
    # If the ellipsoid is not specified, the value of GRS80 is used.
    a, ƒ = GEODETIC_DATUM.get(ellipsoid, GEODETIC_DATUM.get(ELLIPSOID_GRS80))
    b = (1 - ƒ) * a

    # Convert to radians (except distance)
    φ1 = math.radians(lat)
    λ1 = math.radians(lon)
    α1 = math.radians(azimuth)
    s = distance

    sinα1 = math.sin(α1)
    cosα1 = math.cos(α1)

    # Modified latitude (latitude on the auxiliary sphere)
    U1 = math.atan((1 - ƒ) * math.tan(φ1))

    sinU1 = math.sin(U1)
    cosU1 = math.cos(U1)
    tanU1 = math.tan(U1)

    σ1 = math.atan2(tanU1, cosα1)
    sinα = cosU1 * sinα1
    cos2α = 1 - sinα ** 2
    u2 = cos2α * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))

    # Initialize σ with s/(b*A)
    σ = s / (b * A)

    # Iterate the following calculations until σ converges
    # Set an upper limit on the number of iterations, as convergence may not occur at some locations.
    for i in range(ITERATION_LIMIT):
        cos2σm = math.cos(2 * σ1 + σ)
        sinσ = math.sin(σ)
        cosσ = math.cos(σ)
        Δσ = B * sinσ * (cos2σm + B / 4 * (cosσ * (-1 + 2 * cos2σm ** 2) - B / 6 * cos2σm * (-3 + 4 * sinσ ** 2) * (-3 + 4 * cos2σm ** 2)))
        σʹ = σ
        σ = s / (b * A) + Δσ

        # If the deviation is less than .0000000000000001, break
        if abs(σ - σʹ) <= 1e-12:
            break
    else:
        # Returns None if the calculation does not converge
        return None

    # Once σ converges to the desired accuracy, perform the following calculations
    x = sinU1 * sinσ - cosU1 * cosσ * cosα1
    φ2 = math.atan2(sinU1 * cosσ + cosU1 * sinσ * cosα1, (1 - ƒ) * math.sqrt(sinα ** 2 + x ** 2))
    λ = math.atan2(sinσ * sinα1, cosU1 * cosσ - sinU1 * sinσ * cosα1)
    C = ƒ / 16 * cos2α * (4 + ƒ * (4 - 3 * cos2α))
    L = λ - (1 - C) * ƒ * sinα * (σ + C * sinσ * (cos2σm + C * cosσ * (-1 + 2 * cos2σm ** 2)))
    λ2 = L + λ1

    α2 = math.atan2(sinα, -x) + math.pi

    return {
        'lat': math.degrees(φ2),     # Latitude
        'lon': math.degrees(λ2),     # Longitude
        'azimuth': math.degrees(α2), # Azimuth
    }

# Main loop
waypoint_len = len(waypoint_list)
incre = 0
while waypoint_len > 1:
    if waypoint_len >= 3:
        waypoint1_lat = waypoint_list[incre+1][0]
        waypoint1_lon = waypoint_list[incre+1][1]
        waypoint2_lat = waypoint_list[incre+2][0]
        waypoint2_lon = waypoint_list[incre+2][1]
        
        # waypoint1 coordinate calculation
        result = vincenty_inverse(waypoint0_lat,waypoint0_lon,waypoint1_lat,waypoint1_lon)
        long1, rad = result["distance"], result["azimuth1"]
        # azimuth -> polar
        rad = math.radians((90 - rad + 360) % 360)
        x1 = long1 * math.cos(rad)
        y1 = long1 * math.sin(rad)
        vector_1 = np.array([x1,y1])

        # waypoint2 coordinate calculation
        result = vincenty_inverse(waypoint0_lat,waypoint0_lon,waypoint2_lat,waypoint2_lon)
        long2, rad2 = result["distance"], result["azimuth1"]
        rad2 = math.radians((90 - rad2 + 360) % 360)
        x2 = long2 * math.cos(rad2)
        y2 = long2 * math.sin(rad2)
        vector_2 = np.array([x2,y2])

        # waypoint1_2 coordinate calculation
        result = vincenty_inverse(waypoint1_lat,waypoint1_lon,waypoint2_lat,waypoint2_lon)
        long12, rad12 = result["distance"], result["azimuth1"]
        rad12 = math.radians((90 - rad12 + 360) % 360)
        x12 = long12 * math.cos(rad12)
        y12 = long12 * math.sin(rad12)
        vector_12 = np.array([x12,y12])

        A = (math.acos((long1**2+long12**2-long2**2)/(2*long1*long12)))*180/math.pi
        b = r * math.tan(math.radians(90-A/2))
        vector_0d = (vector_1) - (b * (vector_1/np.linalg.norm(vector_1)))
        long_0d = np.sqrt(vector_0d[0] ** 2 + vector_0d[1] ** 2)
        vector_p0d2 = (vector_1) + (b * ((vector_2 - vector_1)/np.linalg.norm(vector_2 - vector_1)))
        long_d2p2 = long12 - b
        vector_N = np.array([[0,1],[-1,0]]) @ vector_1
        vector_d1c = (np.sign(np.cross(vector_1,vector_2)) * (vector_N/np.linalg.norm(vector_N)) * r) * -1
        vector_0c = vector_0d - np.sign(np.cross(vector_1,vector_2)) * (vector_N/np.linalg.norm(vector_N)) * r 

        q,mod = divmod(long_0d,vehicle_velocity)
        count = bak_count
        # If q == 0, pass to the next route generation. Otherwise, the route is generated.
        if q != 0:
            for i in range(int(q)):
                waypoint_xy = (vehicle_velocity + (vehicle_velocity * i)) * (vector_1 / (np.linalg.norm(vector_1)))
                waypoint_x.append(waypoint_xy[0]+offset_x)
                waypoint_y.append(waypoint_xy[1]+offset_y)
                plt.plot(waypoint_x[count+i],waypoint_y[count+i],marker='.',markersize="12")
            count = int(count + q)

        vector_d1d2 = vector_p0d2 - vector_0d
        long_d1d2 = np.sqrt((vector_p0d2[0] - vector_0d[0]) ** 2 + (vector_p0d2[1] - vector_0d[1]) ** 2)
        circle_theta = math.acos((r**2+r**2-long_d1d2**2)/(2*r*r)) * (180/math.pi)
        arc_d1d2 = 2 * math.pi * r * (circle_theta/360)
        bak_mod = mod
        q,mod = divmod(arc_d1d2+bak_mod,vehicle_velocity)
        # If q == 0, pass to the next route generation. Otherwise, the route is generated.
        if q != 0:
            for i in range(int(q)):
                theta = (((vehicle_velocity - bak_mod)+vehicle_velocity * i)/ (2 * math.pi * r)) * 360
                al = math.sqrt(r**2+r**2-2*r**2*(math.cos(math.radians(theta))))
                fai = math.acos((al**2+r**2-r**2)/(2*al*r)) * np.sign(np.cross(vector_d1c,vector_d1d2))
                vector_nn = np.array([[math.cos(fai),-1*math.sin(fai)],[math.sin(fai),math.cos(fai)]]) @ vector_d1c
                vector_0circle = vector_0d + ((vector_nn/np.linalg.norm(vector_nn)) * al)
                waypoint_x.append(vector_0circle[0]+offset_x)
                waypoint_y.append(vector_0circle[1]+offset_y)
                plt.plot(waypoint_x[count+i],waypoint_y[count+i],marker='.',markersize="12")
            count = int(count + q)
        # If there are three waypoints, the route is generated until the end. 
        # if there are four or more waypoints, only one point is taken and the next process is performed.
        if waypoint_len == 3:
            #waypoint1to2
            bak_mod = mod
            q,mod = divmod(long_d2p2+bak_mod,vehicle_velocity)
            for i in range(int(q)):
                waypoint_xy = vector_p0d2 + ((vehicle_velocity - bak_mod)+(vehicle_velocity * i)) * (vector_12 / (np.linalg.norm(vector_12)))
                waypoint_x.append(waypoint_xy[0]+offset_x)
                waypoint_y.append(waypoint_xy[1]+offset_y)
                plt.plot(waypoint_x[count+i],waypoint_y[count+i],marker='.',markersize="12")
            count = int(count+q)
            #goal
            waypoint_x.append(x2+offset_x)
            waypoint_y.append(y2+offset_y)
            plt.plot(waypoint_x[count],waypoint_y[count],marker='*',markersize="15")
            waypoint_len -= 1
        else:
            #waypoint1to2
            bak_mod = mod
            waypoint_xy = vector_p0d2 + ((vehicle_velocity - bak_mod)+(vehicle_velocity * 0)) * (vector_12 / (np.linalg.norm(vector_12)))
            waypoint_x.append(waypoint_xy[0]+offset_x)
            waypoint_y.append(waypoint_xy[1]+offset_y)
            plt.plot(waypoint_x[count],waypoint_y[count],marker='.',markersize="12")
            offset_x = waypoint_x[-1]
            offset_y = waypoint_y[-1]
            count = int(count+1)
        for i in range (count-bak_count):
            way_long = math.sqrt(waypoint_x[bak_count+i]**2+waypoint_y[bak_count+i]**2)
            way_rad = math.atan2(waypoint_y[bak_count+i], waypoint_x[bak_count+i])
            degree = (90 - math.degrees(way_rad) + 360) % 360
            # Reference latitude, longitude, angle, and distance
            result = vincenty_direct(init_lat, init_lon, degree, way_long, 1)
            if result:
                waypoint_lat.append(result["lat"])
                waypoint_lon.append(result["lon"])
                waypoint_ref.append([result["lat"],result["lon"]])
        waypoint0_lat = waypoint_lat[-1]
        waypoint0_lon = waypoint_lon[-1]
        bak_count = count
    else:   # When there are two waypoints
        # Waypoint1 coordinate calculation
        waypoint0_lat = waypoint_list[incre][0]
        waypoint0_lon = waypoint_list[incre][1]
        waypoint1_lat = waypoint_list[incre+1][0]
        waypoint1_lon = waypoint_list[incre+1][1]

        result = vincenty_inverse(waypoint0_lat,waypoint0_lon,waypoint1_lat,waypoint1_lon)
        long1, rad = result["distance"], result["azimuth1"]
        rad = math.radians((90 - rad + 360) % 360)
        x1 = long1 * math.cos(rad)
        y1 = long1 * math.sin(rad)
        vector_1 = np.array([x1,y1])

        q,mod = divmod(long1,vehicle_velocity)
        #If q == 0 then error. Otherwise, the route is generated.
        if q != 0:
            for i in range(int(q)):
                waypoint_xy = (vehicle_velocity + (vehicle_velocity * i)) * (vector_1 / (np.linalg.norm(vector_1)))
                waypoint_x.append(waypoint_xy[0])
                waypoint_y.append(waypoint_xy[1])
                plt.plot(waypoint_x[i],waypoint_y[i],marker='.',markersize="12")
            count = int(q)
        else:
            print("Distance to target is not sufficient")
            sys.exit()

        for i in range (count):
            way_long = math.sqrt(waypoint_x[bak_count+i]**2+waypoint_y[bak_count+i]**2)
            way_rad = math.atan2(waypoint_y[bak_count+i], waypoint_x[bak_count+i])
            degree = (90 - math.degrees(way_rad) + 360) % 360
            # Reference latitude, longitude, angle, and distance
            result = vincenty_direct(init_lat, init_lon, degree, way_long, 1)
            if result:
                waypoint_lat.append(result["lat"])
                waypoint_lon.append(result["lon"])
                waypoint_ref.append([result["lat"],result["lon"]])
    waypoint_len -= 1
    incre += 1

print(waypoint_ref)