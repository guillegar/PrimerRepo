
import pymysql
from datetime import datetime
import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
import pylab as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='', db='bandera')

cur = conn.cursor()
#recojo datos de la base de datos
df_hints = pd.read_sql("SELECT * FROM hints_log", con=conn)
df_scores = pd.read_sql("SELECT * FROM scores_log", con=conn)
df_failures = pd.read_sql("SELECT * FROM failures_log", con=conn)
df_teams = pd.read_sql("SELECT id, points, last_score,created_ts FROM teams", con=conn)
df_teamsData = pd.read_sql("SELECT team_id,name FROM teams_data", con=conn)
df_teamsData=df_teamsData[df_teamsData.name != ""]



   #for  row in df.itertuples(index=True, name='Pandas'):
   #    sec=time.mktime(datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S').timetuple())
   #    df.loc[df.ts == row[2], 'ts'] = sec
   #    df.ts[2]=(sec)
df_hints['ts'] = df_hints['ts'].astype(np.str)
def scores_at_first(df_failures,df_scores,df_result):

    for i, row in df_scores.iterrows():
        time=row['ts']
        level=row['level_id']
        team=row['team_id']
        und=df_failures.loc[(df_failures['ts']<time) & (df_failures['team_id']==(team))& (df_failures['level_id']==(level))]
        if(und.empty):
            df_result.loc[df_result.team_id == team,'scores_at_first']+=1


def tiempo_medio(df_actions,df_result):
    df_actions =df_actions.sort_values('ts')
    for i, row in df_actions.iterrows():
        time=row['ts']
        team=row['team_id']
        #esto es un select
        und=df_result.loc[(df_result['team_id']==(team))]

        if((time-und['ultimo_intento'].iloc[0])<144000):
            df_result.loc[df_result.team_id == team,'media_intentos']+=time-und['ultimo_intento']
            df_result.loc[df_result.team_id == team,'numero_intentos']+=1
        else:
            df_result.loc[df_result.team_id == team,'numero_intentos']+=1

# anadir aqui el numero de sesiones

        df_result.loc[df_result.team_id == team,'ultimo_intento']=time



def ts_a_segundos(df,column_name):

    df[column_name] = df[column_name].astype(np.str)

    for i, row in df.iterrows():
        tiempo=str(row[column_name])
        sec = time.mktime(datetime.strptime(tiempo, '%Y-%m-%d %H:%M:%S').timetuple())
        df.at[i,column_name]=sec

#pasar tiempos a segundos
ts_a_segundos(df_hints,'ts')
ts_a_segundos(df_failures ,'ts')
ts_a_segundos(df_scores ,'ts')
ts_a_segundos(df_teams ,'last_score')
ts_a_segundos(df_teams ,'created_ts')
#tiempo desde que se registro hasta su ultima actividad
df_teams['total_time']=df_teams['last_score']-df_teams['created_ts']
# elimina fecha de creacion y la de la ultima actividad para dejar solo el tiempo total, evitar datos relacionados
df_teams=df_teams.drop(['last_score','created_ts'],axis=1)
df_actions = pd.merge(df_failures, df_scores,how='outer')
df_actions = pd.merge(df_actions, df_hints,how='outer')

df_actions=df_actions.drop(['flag','points','type'],axis=1)

team_countPlayers = df_teamsData.groupby('team_id')['team_id'].count().reset_index(name ='total_players')

team_countFails = df_failures.groupby('team_id')['team_id'].count().reset_index(name ='total_fails')
team_countHints = df_hints.groupby('team_id')['team_id'].count().reset_index(name ='total_hints')
team_countScores = df_scores.groupby('team_id')['team_id'].count().reset_index(name ='total_scores')

df_result = pd.merge(team_countHints, team_countFails,on='team_id',how='outer')
df_result = pd.merge(df_result, team_countScores,on='team_id',how='outer')
df_result = pd.merge(df_result, team_countPlayers,on='team_id',how='outer')

df_teams = df_teams.rename(columns={'id': 'team_id'})
df_result = pd.merge(df_result, df_teams,on='team_id',how='outer')
df_result["scores_at_first"]=0
df_result["numero_intentos"]=0
df_result["sesiones"]=0
df_result["ultimo_intento"]=0
df_result["media_intentos"]=0
scores_at_first(df_failures,df_scores,df_result)
tiempo_medio(df_actions,df_result)
df_result['media_intentos']=df_result['media_intentos']/df_result['numero_intentos']





df_result=df_result.fillna(0)
df_result=df_result[df_result.total_scores != 0]
df_result_data=df_result.copy()
df_teams=df_result.team_id
df_result=df_result.drop(['numero_intentos','team_id',"ultimo_intento","total_time"],axis=1)

# Standardize the data attributes for the Iris dataset.
from sklearn.datasets import load_iris
from sklearn import preprocessing

df_result = preprocessing.normalize(df_result)



pca = PCA(n_components=2).fit(df_result)
pca_2d = pca.transform(df_result)
print pca.get_precision()
print df_result

print pca
kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit(df_result)
print kmeans.cluster_centers_.shape
pl.scatter(pca_2d[:, 0], pca_2d[:, 1])
pl.figure('K-means with 3 clusters')
pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=kmeans.labels_)
for i, txt in enumerate(df_teams):
    pl.annotate(txt,  xy=( pca_2d[i, 0],  pca_2d[i, 1]))
print( kmeans.labels_)
pl.show()

Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(df_result).score(df_result) for i in range(len(kmeans))]
score
pl.plot(Nc,score)
pl.xlabel('Number of   Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()

cur.close()
conn.close()