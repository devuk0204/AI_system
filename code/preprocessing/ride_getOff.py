import pandas as pd

subway = pd.read_csv('/home/devuk/code/AI_system/data/subway_user.csv')

subway['ride'] = 0
subway['get_off'] = 0

for i in range(len(subway)) :
    if subway.type[i] == '승차' :
        subway.ride[i] = 1
    else :
        subway.ride[i] = 0
        
    if subway.type[i] == '하차' :
        subway.get_off[i] = 1
    else :
        subway.get_off[i] = 0
        
subway.to_csv('/home/devuk/code/AI_system/data/subway_user.csv', index = False)