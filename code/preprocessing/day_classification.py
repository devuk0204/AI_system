import pandas as pd

subway = pd.read_csv('/home/devuk/code/AI_system/data/subway_user.csv')

subway['weekday'] = 0
subway['weekend'] = 0

for i in range(len(subway)) :
    if subway.day[i] == '토' or subway.day[i] == '일' :
        subway.weekend[i] = 1
    else :
        subway.weekday[i] = 1
        
subway.to_csv('/home/devuk/code/AI_system/data/subway_user.csv', index = False)