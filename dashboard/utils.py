import psycopg2
import pandas as pd

def load_data(query):
    conn = psycopg2.connect(
        host="localhost",       # CHANGE if needed
        dbname="yourdb",        # CHANGE to your DB name
        user="youruser",        # CHANGE to your username
        password="yourpassword" # CHANGE
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df
