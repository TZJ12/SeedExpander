import pymysql
import hashlib
from typing import List, Dict, Any
from config import Config

class DBManager:
    def __init__(self):
        self.host = Config.DB_HOST
        self.port = Config.DB_PORT
        self.user = Config.DB_USER
        self.password = Config.DB_PASSWORD
        self.db_name = Config.DB_NAME
        self._init_db()

    def _get_connection(self, db=None):
        return pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=db,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

    def _init_db(self):
        """Initialize the database and tables."""
        try:
            # 1. Create Database if not exists
            # Connect without selecting DB to create it
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            conn.commit()
            conn.close()
            
            # 2. Create Tables
            conn = self._get_connection(db=self.db_name)
            cursor = conn.cursor()
            
            # Jailbreak Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS jailbreak_records (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    template_id VARCHAR(255),
                    data_id VARCHAR(255),
                    input_text TEXT,
                    input LONGTEXT,
                    model_name VARCHAR(255),
                    output LONGTEXT,
                    reason LONGTEXT,
                    status VARCHAR(50),
                    category VARCHAR(255),
                    subcategory VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_record (template_id, data_id)
                )
            ''')

            # Failed Table
            # Use input_hash for uniqueness constraint since input (LONGTEXT) is too long for key
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS failed_records (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    template_id VARCHAR(255),
                    data_id VARCHAR(255),
                    input_text TEXT,
                    input LONGTEXT,
                    input_hash VARCHAR(32),
                    model_name VARCHAR(255),
                    output LONGTEXT,
                    reason LONGTEXT,
                    status VARCHAR(50),
                    category VARCHAR(255),
                    subcategory VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_failed (template_id, data_id, input_hash)
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[DB Init Error] {e}")

    def _get_field_from_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fields for database insertion."""
        eval_result = record.get("eval_result", {})
        if not eval_result:
            eval_result = {}
            
        inp = eval_result.get("input") or record.get("rendered") or ""
        # Calculate hash for uniqueness in failed_records
        input_hash = hashlib.md5(inp.encode('utf-8')).hexdigest() if inp else ""

        return {
            "template_id": record.get("template_id"),
            "data_id": str(record.get("data_id")),
            "input_text": record.get("input_text"),
            "input": inp,
            "input_hash": input_hash,
            "model_name": eval_result.get("modelName"),
            "output": eval_result.get("output"),
            "reason": eval_result.get("reason"),
            "status": eval_result.get("status"),
            "category": record.get("category"),
            "subcategory": record.get("subcategory")
        }

    def save_jailbreak_batch(self, records: List[Dict[str, Any]]):
        """Batch insert/ignore jailbreak records."""
        if not records:
            return

        try:
            conn = self._get_connection(db=self.db_name)
            cursor = conn.cursor()
            
            data_to_insert = []
            for r in records:
                fields = self._get_field_from_record(r)
                data_to_insert.append((
                    fields["template_id"],
                    fields["data_id"],
                    fields["input_text"],
                    fields["input"],
                    fields["model_name"],
                    fields["output"],
                    fields["reason"],
                    fields["status"],
                    fields["category"],
                    fields["subcategory"]
                ))

            # MySQL INSERT IGNORE
            sql = '''
                INSERT IGNORE INTO jailbreak_records 
                (template_id, data_id, input_text, input, model_name, output, reason, status, category, subcategory)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            '''
            
            cursor.executemany(sql, data_to_insert)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[DB Save Error - Jailbreak] {e}")

    def save_failed_batch(self, records: List[Dict[str, Any]]):
        """Batch insert/ignore failed records."""
        if not records:
            return

        try:
            conn = self._get_connection(db=self.db_name)
            cursor = conn.cursor()
            
            data_to_insert = []
            for r in records:
                fields = self._get_field_from_record(r)
                data_to_insert.append((
                    fields["template_id"],
                    fields["data_id"],
                    fields["input_text"],
                    fields["input"],
                    fields["input_hash"],
                    fields["model_name"],
                    fields["output"],
                    fields["reason"],
                    fields["status"],
                    fields["category"],
                    fields["subcategory"]
                ))

            # MySQL INSERT IGNORE
            sql = '''
                INSERT IGNORE INTO failed_records 
                (template_id, data_id, input_text, input, input_hash, model_name, output, reason, status, category, subcategory)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            '''
            
            cursor.executemany(sql, data_to_insert)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[DB Save Error - Failed] {e}")

