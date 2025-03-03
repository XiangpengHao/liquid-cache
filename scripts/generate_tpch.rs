use duckdb::{Connection, Result};

/*
   1. generate_tpch.rs
       Loads tpch from duckdb and writes to parquet (tpch.parquet)

   2. Write tester
       Reads parquet file and runs each query

   3. Write checker
       Reads query and runs on duckdb mem-db
*/

const SCALE_FACTOR: i32 = 1;
fn main() -> Result<()> {
    let conn = Connection::open_in_memory()?;

    conn.execute_batch("INSTALL tpch; LOAD tpch")?;
    conn.execute("CALL dbgen(sf=?)", [SCALE_FACTOR])?;

    let mut stmt = conn.prepare("SHOW TABLES;")?;
    let mut rows = stmt.query([])?;

    // Print results
    while let Some(row) = rows.next()? {
        let table_name: String = row.get(0)?;
        println!("{}", table_name);
    }

    return Ok(());
}
