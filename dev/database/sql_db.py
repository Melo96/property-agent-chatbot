import pandas as pd
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    Float,
)

csv_path = '/Users/yangkaiwen/Documents/hypergai-chatbot/data/real_estates_0528.csv'

engine = create_engine("sqlite:///real_estates_0528.db")
metadata_obj = MetaData()

# create real estates table
table_name = "real_estates"
real_estates_table = Table('real_estates', metadata_obj,
    Column('楼盘ID', Integer, primary_key=True),
    Column('楼盘名称', String, primary_key=True),
    Column('楼盘别名', String),
    Column('均价', Float),
    Column('总价', Float),
    Column('省份', String),
    Column('城市', String),
    Column('区域', String),
    Column('销售状态', String),
    Column('车位数量', Integer),
    Column('总栋数', Float),
)
metadata_obj.create_all(engine)

# read csb
usecols = ['楼盘ID', '楼盘名称', '楼盘别名', '均价（元/平）', '总价（套/万元）', '省份', '城市', '区域', '销售状态', '车位数量', '总栋数']
df = pd.read_csv(csv_path, usecols=usecols).drop_duplicates(subset='楼盘名称', keep=False).reset_index(drop=True)
df.rename({'均价（元/平）': '均价', '总价（套/万元）': '总价'}, axis=1, inplace=True)
df.to_sql('real_estates', engine, if_exists='replace', index=False)
