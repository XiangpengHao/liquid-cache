import duckdb
import pyarrow.parquet as pq
from tqdm import tqdm
import pandas as pd
from io import StringIO


SCALE_FACTOR = 1 # Only 0.01, 0.1, and 1 have expected answers

con = duckdb.connect(database=':memory:')
con.execute("INSTALL tpch; LOAD tpch")
con.execute(f"CALL dbgen(sf={SCALE_FACTOR})")
#print(con.execute("show tables").fetchall())


res = con.execute(f"select (query_nr, answer) from tpch_answers() where scale_factor={SCALE_FACTOR}").fetchall()

for row in tqdm(res):
    (query_nr, answer) = row[0]

    data_file = StringIO(answer)
    df = pd.read_csv(data_file, sep="|")

    df.to_parquet(f"tpch_data/answers/{query_nr}.parquet")

table_names = [i[0] for i in con.execute("show tables").fetchall()]

for table in tqdm(table_names):
    con.execute(f"COPY {table} TO 'tpch_data/data/{table}.parquet' (FORMAT PARQUET);")