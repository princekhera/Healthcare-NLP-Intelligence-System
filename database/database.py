from sqlalchemy import create_engine
import pandas as pd

engine = create_engine("sqlite:///database.db")

def store_data(df):
    df.to_sql("abstracts", engine, if_exists="replace", index=False)

def query_data(keyword):
    query = f"""
    SELECT * FROM abstracts
    WHERE abstract LIKE '%{keyword}%'
    LIMIT 5
    """
    return pd.read_sql(query, engine)