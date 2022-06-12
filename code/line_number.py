import pandas as pd

subway = pd.read_csv('/home/devuk/code/AI_system/data/subway_user.csv')

subway['line_number'] = 0

for i in range(len(subway)) :
    if subway.station_number[i] > 0 and subway.station_number[i] < 200 :
        subway.line_number[i] = 1
    elif subway.station_number[i] >= 200 and subway.station_number[i] < 300 :
        subway.line_number[i] = 2
    elif subway.station_number[i] >= 300 and subway.station_number[i] < 400 :
        subway.line_number[i] = 3
    else :
        subway.line_number[i] = 4

subway.to_csv('/home/devuk/code/AI_system/data/subway_user.csv', index = False)