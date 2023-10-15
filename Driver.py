# Databricks notebook source
# DBTITLE 1,Import Libraries
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number,rank
from pyspark.sql import SparkSession
df_dict_output={}
df_dict={}
read_table_list=[]

# COMMAND ----------

# DBTITLE 1,Input Dict
for i in dbutils.fs.ls('/mnt/RAW/FILES/GLOBAL_RC_CSA4/DATASETS'):
  df_dict[i.name.split('.')[0].upper().split('_USE')[0]]=spark.read.format('csv').option('inferschema','true').option('header','true').load(i.path)

# COMMAND ----------

# DBTITLE 1,Q.1 Find the number of crashes (accidents) in which number of persons killed are male
display(df_dict['PRIMARY_PERSON'].filter(F.col('DEATH_CNT')>0).groupBy('PRSN_GNDR_ID').count())

# COMMAND ----------

display(df_dict['PRIMARY_PERSON'].filter(F.col('DEATH_CNT')>0).groupBy('CRASH_ID','PRSN_GNDR_ID').count().groupBy('PRSN_GNDR_ID').count())

# COMMAND ----------

# DBTITLE 1,Q.2 How many two wheelers are booked for crashes? 
print(df_dict['UNITS'].filter(F.col('VEH_BODY_STYL_ID').like('%MOTORCYCLE%')).count())

# COMMAND ----------

# DBTITLE 1,Q.3 Which state has highest number of accidents in which females are involved
display(df_dict['PRIMARY_PERSON'].filter(F.col('PRSN_GNDR_ID')=='FEMALE').groupBy('PRSN_GNDR_ID','DRVR_LIC_STATE_ID').count().orderBy(F.col('count').desc(), F.col('PRSN_GNDR_ID')))

# COMMAND ----------

# DBTITLE 1,Q.4 Which are the Top 5th to 15th VEH_MAKE_IDs that contribute to a largest number of injuries including death
windowspec=Window.orderBy(F.col('count').desc())
display(df_dict['UNITS'].filter((F.col('TOT_INJRY_CNT')>0) | (F.col('DEATH_CNT') > 0)).groupBy('VEH_MAKE_ID').count().withColumn('RANK', F.dense_rank().over(windowspec)).filter(F.col('RANK').between(5,15)))

# COMMAND ----------

# DBTITLE 1,Q5 For all the body styles involved in crashes, mention the top ethnic user group of each unique body style  
windowspec=Window.partitionBy('VEH_BODY_STYL_ID').orderBy(F.col('count').desc())
display(df_dict['PRIMARY_PERSON'].join(df_dict['UNITS'],['CRASH_ID','UNIT_NBR'],'left').select(df_dict['PRIMARY_PERSON']['*'],df_dict['UNITS']['VEH_BODY_STYL_ID']).distinct().groupBy('VEH_BODY_STYL_ID','PRSN_ETHNICITY_ID').count().withColumn('RANK', F.dense_rank().over(windowspec)).filter(F.col('RANK')==1))

# COMMAND ----------

# DBTITLE 1,Q6 Among the crashed cars, what are the Top 5 Zip Codes with highest number crashes with alcohols as the contributing factor to a crash (Use Driver Zip Code)
display(df_dict['PRIMARY_PERSON'].filter(F.col('PRSN_ALC_RSLT_ID')=='Positive').groupBy('DRVR_ZIP').count().orderBy(F.col('count').desc()))

# COMMAND ----------

# DBTITLE 1,Q7 Count of Distinct Crash IDs where No Damaged Property was observed and Damage Level (VEH_DMAG_SCL~) is above 4 and car avails Insurance
display(df_dict['PRIMARY_PERSON'].join(df_dict['UNITS'],['CRASH_ID','UNIT_NBR'],'left').select(df_dict['PRIMARY_PERSON']['*'], df_dict['UNITS']['VEH_DMAG_SCL_1_ID']).withColumn('DAMAGE_LEVEL', F.split(F.col('VEH_DMAG_SCL_1_ID'),' ')[1]).join(df_dict['DAMAGES'],'CRASH_ID','left').filter((F.col('DAMAGED_PROPERTY').isNull()) & (F.col('DAMAGE_LEVEL')>4)).select('CRASH_ID').distinct())

# COMMAND ----------

print(df_dict['PRIMARY_PERSON'].join(df_dict['UNITS'],['CRASH_ID','UNIT_NBR'],'left').select(df_dict['PRIMARY_PERSON']['*'], df_dict['UNITS']['VEH_DMAG_SCL_1_ID']).withColumn('DAMAGE_LEVEL', F.split(F.col('VEH_DMAG_SCL_1_ID'),' ')[1]).join(df_dict['DAMAGES'],'CRASH_ID','left').filter((F.col('DAMAGED_PROPERTY').isNull()) & (F.col('DAMAGE_LEVEL')>4)).select('CRASH_ID').distinct().count())

# COMMAND ----------

# DBTITLE 1,Q8 Determine the Top 5 Vehicle Makes where drivers are charged with speeding related offences, has licensed Drivers, used top 10 used vehicle colours and has car licensed with the Top 25 states with highest number of offences (to be deduced from the data)
windowspec=Window.orderBy(F.col('count').desc())
colour=df_dict['UNITS'].groupBy('VEH_COLOR_ID').count().withColumn('RANK', F.dense_rank().over(windowspec)).filter(F.col('RANK')<=10).select('VEH_COLOR_ID').rdd.flatMap(lambda x:x).collect()
state=df_dict['UNITS'].groupBy('VEH_LIC_STATE_ID').count().withColumn('RANK', F.dense_rank().over(windowspec)).filter(F.col('RANK')<=25).select('VEH_LIC_STATE_ID').rdd.flatMap(lambda x:x).collect()
display(df_dict['UNITS'].filter((F.col('VEH_COLOR_ID').isin(colour)) & (F.col('VEH_LIC_STATE_ID').isin(state)) & ((F.col('CONTRIB_FACTR_2_ID').like('%SPEED%')) | (F.col('CONTRIB_FACTR_1_ID').like('%SPEED%')) | (F.col('CONTRIB_FACTR_P1_ID').like('%SPEED%')) )).join(df_dict['PRIMARY_PERSON'],['CRASH_ID','UNIT_NBR'],'left').select(df_dict['UNITS']['*'], df_dict['PRIMARY_PERSON']['DRVR_LIC_TYPE_ID']).distinct().filter(F.col('DRVR_LIC_TYPE_ID').like('%DRIVER%')).groupBy('VEH_MAKE_ID').count().withColumn('RANK', F.dense_rank().over(windowspec)).filter(F.col('RANK')<=5))
