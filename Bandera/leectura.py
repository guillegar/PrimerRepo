
import pymysql
from datetime import datetime
import time
import numpy as np
import pandas as pd

conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='', db='bandera')

cur = conn.cursor()

df_hints = pd.read_sql("SELECT * FROM hints_log", con=conn)
df_activity = pd.read_sql("SELECT * FROM activity_log", con=conn)
df_scores = pd.read_sql("SELECT * FROM scores_log", con=conn)
df_failures = pd.read_sql("SELECT * FROM failures_log", con=conn)



   #for  row in df.itertuples(index=True, name='Pandas'):
   #    sec=time.mktime(datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S').timetuple())
   #    df.loc[df.ts == row[2], 'ts'] = sec
   #    df.ts[2]=(sec)
print(df_hints.dtypes)
df_hints['ts'] = df_hints['ts'].astype(np.str)
print(df_hints.dtypes)


def ts_a_segundos(df,column_name):
    print(df.dtypes)
    df[column_name] = df[column_name].astype(np.str)
    print(df.dtypes)
    for i, row in df.iterrows():
        tiempo=str(row[column_name])
        sec = time.mktime(datetime.strptime(tiempo, '%Y-%m-%d %H:%M:%S').timetuple())
        df.at[i,column_name]=sec


ts_a_segundos(df_hints,'ts')
ts_a_segundos(df_activity ,'ts')
ts_a_segundos(df_failures ,'ts')
ts_a_segundos(df_scores ,'ts')


print df_hints.head
print df_hints.head
print df_hints.head
cur.close()
conn.close()