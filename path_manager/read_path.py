import os
import sqlite3

db_file = 'file_paths.db'

path_file_1 = "/Users/eugen/Downloads/pair_plot.png"
path_file_2 = "/Users/eugen/Downloads/line_plot.png"
path_file_3 = "/Users/eugen/Desktop/api.rtf"

conn = sqlite3.connect(db_file)
cursor = conn.cursor()

cursor.execute('''
        CREATE TABLE IF NOT EXISTS file_paths (
        id INTEGER PRIMARY KEY,
        path TEXT)
        ''')
cursor.execute('SELECT path FROM file_paths WHERE id = 1')
row = cursor.fetchone()

if row:
    stored_path = row[0]
    print(f'Read from database: {stored_path}')
else:
    cursor.execute('INSERT INTO file_paths (id, path) VALUES (?, ?)', (1, path_file_1))
    conn.commit()
conn.close()

