import pandas as pd

subway = pd.read_csv('/home/devuk/code/AI_system/data/subway_user.csv')

subway_temp = subway.drop(['station_number', 'station_name', 'day', 'type'], axis = 1)

subway_temp.to_csv('/home/devuk/code/AI_system/data/feature_target.csv', index = False)