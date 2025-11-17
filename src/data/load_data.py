import sqlite3
import pandas as pd
from typing import Optional

def load_sqlite(db_path: str, table_name: str, where: Optional[str] = None) -> pd.DataFrame:
    query = f"SELECT * FROM {table_name}" + (f" WHERE {where}" if where else "")
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    return df
