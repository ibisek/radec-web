import os

DB_HOST = '10.8.0.30'   # radec
DB_PORT = 3306
DB_NAME = 'radec'
DB_USER = '**'
DB_PASSWORD = '**'
dbConnectionInfo = (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)

if DB_PASSWORD != '**':
    os.environ.setdefault('DB_HOST', DB_HOST)
    os.environ.setdefault('DB_PORT', str(DB_PORT))
    os.environ.setdefault('DB_NAME', DB_NAME)
    os.environ.setdefault('DB_USER', DB_USER)
    os.environ.setdefault('DB_PASSWORD', DB_PASSWORD)

INFLUX_DB_NAME = DB_NAME
INFLUX_DB_HOST = DB_HOST

SQLALCHEMY_DB_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
