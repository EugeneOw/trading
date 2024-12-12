import pickle
import logging
import sqlite3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)


class DBManager:
    def __init__(self, constants_var):
        self.db_file = constants_var


class FilePathManager(DBManager):
    def __init__(self, db_file):
        super().__init__(db_file)

    def fetch_file_path(self, file_id):
        """
        Attempts to fetch file path from database. But creates if it doesn't exist
        :param: file_id: Contains the index of the file's path in the database rows.
        :type: file_id: int

        :return: stored_path
        :rtype: str
        """
        with sqlite3.connect(self.db_file) as conn:  # Using 'with' ensures the connection is closed properly
            cursor = conn.cursor()
            cursor.execute('SELECT path FROM file_paths WHERE id = ?', (file_id,))
            result = cursor.fetchone()
            if result is None:
                return None
            return result[0]


class QTableManager(DBManager):
    def __init__(self, db_file):
        super().__init__(db_file)

    def create_table(self):
        """Creates table if doesn't exists"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS q_table (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                q_table BLOB)
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                ''')
            conn.commit()

    def q_table_operation(self, q_table):
        """Read Q-table from the database, or insert/update accordingly."""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT q_table FROM q_table ORDER BY id DESC LIMIT 1')
            result = cursor.fetchone()
            serialized_q_table = pickle.dumps(q_table)
            serialized_q_table = (serialized_q_table,)
            if result:
                self.update_q_table(serialized_q_table)
            else:
                self.create_table()
                self.insert_q_table(serialized_q_table)
            conn.commit()

    def add_timestamp_column(self):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                    ALTER TABLE q_table ADD COLUMN timestamp DATETIME''')
            cursor.execute('''
                    UPDATE q_table
                    SET timestamp = CURRENT_TIMESTAMP
                    WHERE timestamp IS NULL''')
            conn.commit()

    def update_q_table(self, serialized_q_table):
        """Updates q_table (database) with updated q-table
        :param serialized_q_table: serialized q-table
        :return None
        """

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''UPDATE q_table 
                           SET q_table = ?, timestamp = CURRENT_TIMESTAMP
                           WHERE id = (SELECT MAX(id) FROM q_table)
                           ''', serialized_q_table)
            conn.commit()

    def insert_q_table(self, serialized_q_table):
        """
        Inserts q-table into q_table (database)
        :param serialized_q_table: serialized q-table
        :return: None
        """
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO q_table (q_table) VALUES (?)',serialized_q_table)
            conn.commit()

    def read_q_table(self):
        """
        Reads 'q_table'.db and returns the most updated q-table
        :return: Most updated q-table
        """
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT q_table FROM q_table ORDER BY id DESC LIMIT 1')
            result = cursor.fetchone()
            if result:
                serialized_q_table = result[0]
                return pickle.loads(serialized_q_table)
