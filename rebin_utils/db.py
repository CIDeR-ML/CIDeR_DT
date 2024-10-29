import sqlite3, time, os
import numpy as np
import pandas as pd

from contextlib import closing
from tqdm import tqdm

class TableNotFoundError(Exception):
    pass
class ColumnNotFoundError(Exception):
    pass

class rebin_db:
    def __init__(self, config_file, config_gen):
        self.config_file = config_file
        self.config_gen = config_gen
        self.config = None
        self.conn = None
        self.cursor = None
        self.db_name = None
        self.h5_name = None
        self.h5_file = None
        self.h5_dset = None
        self.h5_dset_name = None
        self.h5_dset_shape = None
        self.h5_dset_dtype = None
        self.h5_dset_chunk = None
        self.h5_dset_compression = None
        self.h5_dset_compression_opts = None
        self.h5_dset_shuffle = None
        self.h5_dset_fletcher32 = None
        self.h5_dset_fill_value = None
        self.h5_dset_scaleoffset = None
        self.h5_dset_complevel = None
        self.h5_dset_fletcher32 = None
        self.h5_dset_filter_opts = None
        self.h5_dset_filter


    def create_db(self, pts, dirs):
        start_time = time.time()
        if os.path.exists(self.dbfile):
            return
        conn = sqlite3.connect(self.dbfile)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS bins
                     (id INTEGER, x REAL, y REAL, z REAL, phi REAL, theta REAL, stats INTEGER, Qmax REAL)''')

        insert_data = []
        for i, p in enumerate(pts):
            for j, d in enumerate(dirs):
                insert_data.append((i * len(dirs) + j, p[0], p[1], p[2], d[1], d[0], 0, 0))
        c.executemany("INSERT INTO bins VALUES (?,?,?,?,?,?,?,?)", insert_data)
        conn.commit()
        conn.close()
        print(f"Took {time.time()-start_time:.2f} sec creating database: {self.dbfile}.")