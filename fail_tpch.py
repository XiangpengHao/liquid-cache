import duckdb
import pandas as pd
from io import StringIO

SCALE_FACTOR = 1 # Only 0.01, 0.1, and 1 have expected answers

# Connect to duckdb, install tpch data
con = duckdb.connect(database=':memory:')
con.execute("INSTALL tpch; LOAD tpch")
con.execute(f"CALL dbgen(sf={SCALE_FACTOR})")


# Query 2 from TPC-H
query = """
select
    s_acctbal,
    s_name,
    n_name,
    p_partkey,
    p_mfgr,
    s_address,
    s_phone,
    s_comment
from
    part,
    supplier,
    partsupp,
    nation,
    region
where
        p_partkey = ps_partkey
  and s_suppkey = ps_suppkey
  and p_size = 15
  and p_type like '%BRASS'
  and s_nationkey = n_nationkey
  and n_regionkey = r_regionkey
  and r_name = 'EUROPE'
  and ps_supplycost = (
    select
        min(ps_supplycost)
    from
        partsupp,
        supplier,
        nation,
        region
    where
            p_partkey = ps_partkey
      and s_suppkey = ps_suppkey
      and s_nationkey = n_nationkey
      and n_regionkey = r_regionkey
      and r_name = 'EUROPE'
)
order by
    s_acctbal desc,
    n_name,
    s_name,
    p_partkey;
"""

# Run query on data
query_results = con.execute(query).fetchall()
print(len(res))

query_num_rows = len(query_results)


# Get baseline (answer) data
answer_results = con.execute(f"select answer from tpch_answers() where scale_factor={SCALE_FACTOR} and query_nr=2").fetchall()
answer_results = answer_results[0][0]

answer_results = pd.read_csv(StringIO(answer_results), sep="|")

answer_num_rows = len(answer_results)

# These should have the same number of rows
assert answer_num_rows == query_num_rows, "Query did not yield expected results (num rows not equal)"