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

csv_path = '/Users/yangkaiwen/Documents/hypergai-chatbot-data/real_estates_0528.csv'

engine = create_engine("sqlite:///real_estates_0528.db")
metadata_obj = MetaData()

# create real estates table
table_name = "real_estates"
real_estates_table = Table('real_estates', metadata_obj,
    Column('楼盘ID', Integer, primary_key=True),
    Column('楼盘名称', String),
    Column('楼盘别名', String),
    Column('均价（元/平）', Float),
    Column('总价（套/万元）', Float),
    Column('省份', String),
    Column('城市', String),
    Column('区域', String),
    Column('销售状态', String),
    Column('特色', String),
    Column('房型', String),
    Column('装修状态', String),
    Column('地址', String),
    Column('物业类型', String),
    Column('产权年限', String),
    Column('容积率', Float),
    Column('绿化率', Float),
    Column('建筑面积', Float),
    Column('占地面积', Float),
    Column('开发商', String),
    Column('预售证', String),
    Column('经度', Float),
    Column('纬度', Float),
    Column('户数', Integer),
    Column('物业费（元/平）', Float),
    Column('物业公司', String),
    Column('开盘时间', String),
    Column('交付时间', String),
    Column('主力户型', String),
    Column('车位数量', Integer),  # This could be a complex type in reality
    Column('总栋数', Float),
    Column('楼层总高', String),  # This could also be a complex type
    Column('电梯户数', String),
    Column('在售楼栋', String),
    Column('售楼处', String),
    Column('楼层状况', String)
)
metadata_obj.create_all(engine)

# read csb
usecols = ['楼盘ID', '楼盘名称', '楼盘别名', '均价（元/平）', '总价（套/万元）', '省份', '城市', '区域', '销售状态', '特色', '房型', '装修状态', '地址', '物业类型', '产权年限', '容积率', '绿化率', '建筑面积', '占地面积', '开发商', '预售证', '经度', '纬度', '户数', '物业费（元/平）', '物业公司', '开盘时间', '交付时间', '主力户型', '车位数量', '总栋数', '楼层总高', '电梯户数', '在售楼栋', '售楼处', '楼层状况']
df = pd.read_csv(csv_path, usecols=usecols).drop_duplicates(subset='楼盘ID', keep=False).reset_index(drop=True)
# df.rename({'均价（元/平）': '均价', '总价（套/万元）': '总价'}, axis=1, inplace=True)
df.to_sql('real_estates', engine, if_exists='replace', index=False)
