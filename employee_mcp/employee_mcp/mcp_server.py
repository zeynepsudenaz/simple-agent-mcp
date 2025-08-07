#!/usr/bin/env python3
from mcp.server.fastmcp import FastMCP
import sqlite3
from typing import Optional
import json

DB_PATH = "C:\\Users\\zeyne\\employee_mcp\\employee_mcp\\employee.db"
mcp = FastMCP("EmployeeDBServer")

@mcp.tool("run_sql", description="Her türlü SQL sorgusu çalıştırır")
def run_sql(query: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    conn.close()
    return [dict(zip(cols, row)) for row in rows]

@mcp.tool("highest_salary_department", description="En yüksek maaşlı departmanı getirir")
def highest_salary_department():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT d.dept_name, MAX(s.amount) AS max_salary
        FROM salary s
        JOIN dept_emp de ON s.emp_no = de.emp_no
        JOIN department d ON de.dept_no = d.dept_no
        GROUP BY d.dept_name
        ORDER BY max_salary DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    conn.close()
    if row:
        return {"department": row[0], "max_salary": row[1]}
    return {"error": "Veri bulunamadı"}

@mcp.tool("get_people", description="Belirtilen departmandaki çalışanları getirir")
def get_people(department: str, limit: Optional[int] = None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    query = """
    SELECT e.emp_no, e.first_name || ' ' || e.last_name AS name, d.dept_name
    FROM employee e
    JOIN dept_emp de ON e.emp_no = de.emp_no
    JOIN department d ON de.dept_no = d.dept_no
    WHERE LOWER(d.dept_name) = LOWER(?)
    """
    params = [department]

    if limit is not None:
        try:
            query += " LIMIT ?"
            params.append(int(limit))
        except ValueError:
            pass  # limit sayı değilse görmezden gel

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    
    if not rows:
        return json.dumps({"error": f"{department} departmanında çalışan bulunamadı."})
    
    result = [
        {"id": row[0], "name": row[1], "department": row[2]}
        for row in rows
    ]
    return json.dumps(result, ensure_ascii=False)  # Türkçe karakter desteği

if __name__ == "__main__":
    print(" FastMCP server başlatılıyor...")
    mcp.run()

if __name__ == "__main__":
    mcp.run()
