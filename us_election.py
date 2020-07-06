#web scraping using pandas in jupyter lab
#US elections

import pandas as pd

url = 'https://en.wikipedia.org/wiki/Elections_in_the_United_States'
data = pd.read_html(url)

#total tables length 33
tables = len(data)  

#displaying each table's columns to check for specific table
for d in data:
    display(d.head(0))  #keeping it short
    
#double check which specific table     
for n in range(tables):
    display(data[n].columns)  #multiIndex

#US election comparison table
data[5].columns
df = data[5]  
df

length = len(df)
year = df.columns

#find out the next presidential election(Kanye West or Donald Trump)
def nextPresElect():
    for y in year:
        yr = df.loc[1, y]
        if yr == 'Yes':
            print('Presidential election is year ' + y)
    
nextPresElect()


#find out the next gubernatorial election in NY state
def nyGovElect():
    for y in year:
        yr = df.loc[4, y]
        if yr.find('NY') >= 1:
            print('Next Gubernatorial election in NY is ',y)

nyGovElect()        




clear
