import pandas as pd

subway = pd.read_csv('/home/devuk/code/AI_system/data/subway_user.csv')

subway['transfer_station'] = 0

transfer_station_code = [119, 219, 123, 305, 124, 125, 402, 205, 208, 227, 233, 313, 306, 309, 317]

for i in range(len(subway)) :
    if subway.station_number[i] in transfer_station_code :
        subway.transfer_station[i] = 1
    else :
        subway.transfer_station[i] = 0

subway.to_csv('/home/devuk/code/AI_system/data/subway_user.csv', index = False)