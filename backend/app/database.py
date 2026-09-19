import os
from pathlib import Path
import psycopg
from psycopg.rows import dict_row


def connect():
    return psycopg.connect(os.environ['DATABASE_URL'], row_factory=dict_row, connect_timeout=5)


def migrate():
    folder = Path(__file__).resolve().parents[1] / 'migrations'
    with connect() as db:
        for path in sorted(folder.glob('*.sql')):
            db.execute(path.read_text())
