# GPS based path planning (Python3)

## Overview
Generates a path for a series of points using latitude and longitude.  
The path consists of straight lines and arcs.  
A route can be generated from two or more waypoints.  
The route is output as a series of points list of latitude and longitude.
![スクリーンショット 2022-06-02 203917](https://user-images.githubusercontent.com/86940058/174790821-a73dfe9a-929a-4e9f-a156-1816e00a570a.png)
### Program Description
GPS_pathplan.py : The generated routes are displayed using matplotlib.  
GPS_pathplan_kml.py : Saves the generated route to a kml file. The saved kml file can be displayed in Google Earth.  

## Requirement
* Python 3.7.
* math
* matplotlib.pyplot
* numpy
* pandas
* sys
* simplekml

## How to use
### Input variable
Set the following variables to values of your choice.  

1. r : Radius of arc path [m]  
2. vehicle_velocity : Interval of a series of points [m] ( Vehicle speed [m/s] )  
3. waypoint*_lat : Latitude of target waypoint [°]   
4. waypoint*_lon : Longitude of target waypoint [°]  
    3.4.waypoint:The first is the initial waypoint of the robot. The last is the target waypoint.
        The coordinates between them specify the waypoint of the turn on the travel route.  

### Output variable
The following list is output.  

1. waypoint_list : Generated route (list of latitude and longitude) [°]  
2. waypoint_lat : Generated route (list of latitude) [°]  
3. waypoint_lat : Generated route (list of longitude) [°]  

## Reference
[[Python]緯度経度から2地点間の距離と方位角を計算する](https://qiita.com/r-fuji/items/99ca549b963cedc106ab)  
[[Python] 始点の緯度経度と方位角と距離から、終点の緯度経度と方位角を求める](https://qiita.com/r-fuji/items/5eefb451cf7113f1e51b)

## Author
happylyrics  
[Twitter](https://twitter.com/happylyr1cs)  
## Licence
MIT
