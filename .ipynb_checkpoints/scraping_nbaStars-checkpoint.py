## Scraping 7 NBA players stats using Pandas

import pandas as pd

urls = ["https://en.wikipedia.org/wiki/Shaquille_O'Neal",
        'https://en.wikipedia.org/wiki/Larry_Bird',
        'https://en.wikipedia.org/wiki/Michael_Jordan',
        'https://en.wikipedia.org/wiki/Kobe_Bryant',
        'https://en.wikipedia.org/wiki/Lebron_James',
        'https://en.wikipedia.org/wiki/Stephen_Curry',
        'https://en.wikipedia.org/wiki/Kevin_Durant']


# check/Test Shaq

pd.read_html(urls[0])[4].set_index('Year')


# scrape by finding the tables, instead of css or html hunt
# 0-4 shaq, 1-4 bird, 2-5 jordan, 3-3 bryant, 4-3 james, 5-3 curry, 6-3 durant

df1 = pd.read_html(urls[0])[4].set_index('Year') # 'shaq'
df2 = pd.read_html(urls[1])[4].set_index('Year') # 'bird'
df3 = pd.read_html(urls[2])[5].set_index('Year') # 'jordan'
df4 = pd.read_html(urls[3])[3].set_index('Year') # 'bryant'
df5 = pd.read_html(urls[4])[3].set_index('Year') # 'james'
df6 = pd.read_html(urls[5])[3].set_index('Year') # 'curry'
df7 = pd.read_html(urls[6])[3].set_index('Year') # 'durant'

df_all = [df1, df2, df3, df4, df5, df6, df7]


# cleaning data
# .replace ('\*', ' ', regex=True) 


def clean_one():
    
    for df in df_all:
        
        df['PPG'] = df['PPG'].replace('\*','',regex=True).astype(float)

        df['MPG'] = df['MPG'].replace('\*','',regex=True).astype(float)

        df['FT%'] = df['FT%'].replace('\*','',regex=True).astype(float)

        df['SPG'] = df['SPG'].replace('\*','',regex=True).astype(float)

        df['FG%'] = df['FG%'].replace('\*','',regex=True).astype(float)
        
        df2['3P%'] = df2['3P%'].replace("\.\.\.",'.376',regex=True).astype(float)  # replacing with avg .376
        
    return

clean_one()


# renaming columns

columns = ['team','games_played','games_started','minutes_perGame','field_goal_percent','3pts_percent'\
        ,'free_throw_percent','rebounds_perGame','assists_perGame','steals_perGame','blocks_perGame','points_perGame']

for df in df_all:
    df.columns = [columns]
    display(df)
    
    
# str = '{}'
# for name in names:
#     columns = [str.format(name),'games_played','games_started','minutes_perGame','field_goal_percent','3pts_percent'\
#         ,'free_throw_percent','rebounds_perGame','assists_perGame','steals_perGame','blocks_perGame','points_perGame']
#     print(columns)


# check Shaq's ppg

df1[['points_perGame']]


for df in df_all:
    display(df.loc['Career','points_perGame'])
    
    
# check data for any asterisk*

for df in df_all:
    print(df.loc['Career','minutes_perGame'].astype(float))
    
    
for df in df_all:
    display(df)
    
    
# remove col 'team', & rows 'career', & 'all-star'

for df in df_all:
    df.drop(['team'], axis=1, level=0, inplace=True, errors='ignore')
    
for df in df_all:
    df.drop(['Career', 'All-Star'], axis=0, inplace=True, errors='ignore')

    display(df)
    

names = ["Shaquille O'Neal",'Larry Bird','Michael Jordan','Kobe Bryant','Lebron James'\
         ,'Stephen Curry','Kevin Durant']

new_dfs = [[names[0],df1],[names[1],df2],[names[2],df3],[names[3],df4]\
             ,[names[4],df5],[names[5],df6],[names[6],df7]]


for n,df in new_dfs:
    display(n,df)
    

for n,df in new_dfs:
    display(n, df[['games_played','points_perGame']])
    

# displays single stats(describe)

for n,df in new_dfs:
    
    print(n, df.describe().loc['mean','points_perGame'])
    

# another method

for n,df in new_dfs:
    
    print(n, df[['points_perGame']].mean())
    

for n,df in new_dfs:
    display(n, df.describe().loc[['mean']])
    

for n,df in new_dfs:
    display(n, df.describe().loc[['mean','max']])
    

frames = [df1, df2, df3, df4, df5, df6, df7]
nba_stars = pd.concat(frames, keys=["Shaquille O'Neal",'Larry Bird','Michael Jordan','Kobe Bryant','Lebron James','Stephen Curry','Kevin Durant'])

pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', None)

nba_stars





clear