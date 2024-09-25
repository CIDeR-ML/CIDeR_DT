import numpy as np
import h5py
import sqlite3
import yaml
import glob
import os
import math
import time

from datetime import datetime
from contextlib import closing
from collections import defaultdict

from wcprod.utils import voxels, directions

class wc_binning():
    def __init__(self, cfg=None, cfg_gen=None):
        self.rows = None
        if cfg: self.configure(cfg)
        if cfg_gen:
            if os.path.isfile(cfg_gen):
                cfg_gen = yaml.safe_load(open(cfg_gen, 'r'))
            else:
                cfg_gen = yaml.safe_load(cfg_gen)
            assert type(cfg_gen) is dict
            self.r0_vox = float(cfg_gen['r0'])
            self.r1_vox = float(cfg_gen['r1'])
            self.z0_vox = float(cfg_gen['z0'])
            self.z1_vox = float(cfg_gen['z1'])
            self.phi0_vox = float(cfg_gen['phi0'])
            self.phi1_vox = float(cfg_gen['phi1'])

    def configure(self, cfg):
        if type(cfg) == str:
            if os.path.isfile(cfg):
                cfg = yaml.safe_load(open(cfg, 'r'))
            else:
                cfg = yaml.safe_load(cfg)
        assert type(cfg) == dict

        self.rmax = float(cfg['Position']['rmax'])
        self.zmax = float(cfg['Position']['zmax'])
        self.gap_space = float(cfg['Position']['gap_space'])
        self.gap_angle = float(cfg['Direction']['gap_angle'])
        self.n_phi_start = int(cfg['Position'].get('n_bins_phi0', 0))
        self.dset_names = cfg['Data']['dset_names']

        self.data = {}
        self.infile = cfg['Data']['input_file']
        self.outfile = cfg['Data']['output_file']
        assert os.path.isfile(self.infile), 'Input file does not exist.'

        self.dbfile = cfg['Database']['db_file']
        self.num_shards = cfg['Database'].get('num_shards', 100)
        self._conn = sqlite3.connect(self.dbfile)

        self.wall_cut = cfg['Action']['wall_cut']
        self.towall_cut = cfg['Action']['towall_cut']

        self.npmt = cfg['Detector']['npmt']

        self.dset = {}
        self.cmprs = cfg['Format'].get('compression', 'gzip')
        self.cmprs_opt = cfg['Format'].get('compression_opts', 5)
        self.drop_unhit = cfg['Format'].get('drop_unhit', False)

    def load_data(self):
        # Load the data from the h5 file
        with h5py.File(self.infile, 'r') as hf:
            for key in self.dset_names:
                self.data[key] = hf[key][:]

    def create_grid(self):
        vox, pts = voxels(-self.zmax, self.zmax, 0, self.rmax, self.gap_space, self.n_phi_start)
        dirs = directions(self.gap_angle)
        #vox: 0:2 r-range, 2:4 phi-range, 4:6 z-range
        return vox, pts, dirs

    def create_sharded_db(self, pts, dirs):
        start_time = time.time()
        with closing(self._conn.cursor()) as c:
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='shard_info'")
            result = c.fetchall()
            if len(result) < 1:
                c.execute('''CREATE TABLE IF NOT EXISTS shard_info
                             (shard_id INTEGER PRIMARY KEY, min_id INTEGER, max_id INTEGER)''')

                for shard in range(self.num_shards):
                    c.execute(f'''CREATE TABLE IF NOT EXISTS bins_{shard}
                             (id INTEGER, x REAL, y REAL, z REAL, phi REAL, theta REAL, stats INTEGER, Qmax REAL)''')

                total_rows = len(pts) * len(dirs)
                rows_per_shard = math.ceil(total_rows / self.num_shards)

                insert_data = [[] for _ in range(self.num_shards)]
                shard_info = []

                for i, p in enumerate(pts):
                    for j, d in enumerate(dirs):
                        row_id = i * len(dirs) + j
                        shard_id = row_id // rows_per_shard
                        insert_data[shard_id].append((row_id, p[0], p[1], p[2], d[1], d[0], 0, 0))

                for shard in range(self.num_shards):
                    if insert_data[shard]:
                        min_id = insert_data[shard][0][0]
                        max_id = insert_data[shard][-1][0]
                        shard_info.append((shard, min_id, max_id))
                        c.executemany(f"INSERT INTO bins_{shard} VALUES (?,?,?,?,?,?,?,?)", insert_data[shard])

                c.executemany("INSERT OR REPLACE INTO shard_info VALUES (?,?,?)", shard_info)
                self._conn.commit()
        print(f"Took {time.time()-start_time:.2f} sec creating database: {self.dbfile}.")

    def create_final_h5(self):
        print("ouput file:", self.outfile)
        if os.path.exists(self.outfile):
            os.remove(self.outfile)
        self.fh5 = h5py.File(self.outfile, 'w')
        self.fh5.attrs['timestamp'] = str(datetime.now())

    def select_bins_from_db(self):
        start_time = time.time()
        self._conn.create_function("SQRT",  1, math.sqrt)
        self._conn.create_function("POW",   2, math.pow)
        self._conn.create_function("ABS",   1, math.fabs)
        self._conn.create_function("ATAN2", 2, math.atan2)

        query = '''
        SELECT id, x, y, z, phi, theta, stats, Qmax
        FROM bins_{shard}
        WHERE (
        (SQRT(POW(x,2)+POW(y,2)) BETWEEN ? AND ?)
        AND 
        ((ATAN2(y, x)*180./? + 360.) % 360. BETWEEN ? AND ?)
        AND
        (z BETWEEN ? AND ?))
        ;
        '''

        with closing(self._conn.cursor()) as c:
            all_rows = []
            for shard in range(self.num_shards):
                c.execute(f"SELECT name FROM sqlite_master where type='index' AND name = 'idx_bins_{shard}_x_y_z'")
                if not c.fetchone():
                    c.execute(f'CREATE INDEX idx_bins_{shard}_x_y_z ON bins_{shard} (x, y, z)')

                shard_query = query.format(shard=shard)
                c.execute(shard_query, (self.r0_vox, self.r1_vox,
                                        math.pi, self.phi0_vox, self.phi1_vox,
                                        self.z0_vox, self.z1_vox))
                shard_rows = c.fetchall()
                all_rows.extend(shard_rows)

        self.rows = np.array(all_rows)
        print(f"Took {time.time()-start_time:.2f} sec to fetch {len(self.rows)} bins that covers the generated voxel from db.")

    def update_sharded_db(self, bin_stats, bin_Qmax):
        start_time = time.time()
        sql_update_query = """UPDATE bins_{shard}
                              SET stats = ?, Qmax = ?
                              WHERE id = ?"""
        shard_update_data = defaultdict(list)
        with closing(self._conn.cursor()) as c:
            # Optimize SQLite for bulk updates
            c.execute("PRAGMA journal_mode = WAL")
            c.execute("PRAGMA synchronous = NORMAL")
            c.execute("PRAGMA cache_size = -64000")  # 64MB cache
            self._conn.execute("BEGIN TRANSACTION")
            for i, r in enumerate(self.rows):
                row_id = r[0]
                c.execute("SELECT shard_id FROM shard_info WHERE ? BETWEEN min_id and max_id", (row_id,))
                shard_id = c.fetchone()[0]
                #check if there is any update for this row
                c.execute(f"SELECT stats, Qmax FROM bins_{shard_id} WHERE id = ?", (row_id,))
                new_stats, new_qmax = list(c.fetchall()[0])
                stats_value = bin_stats[i] + max(r[-2], new_stats)  # Compute stats
                Qmax_value = max(bin_Qmax[i], max(r[-1], new_qmax))  # Compute Qmax
                shard_update_data[shard_id].append((stats_value, Qmax_value, row_id))

            try:
                for shard in range(self.num_shards):
                    if shard in shard_update_data:
                        shard_query = sql_update_query.format(shard=shard)
                        c.executemany(shard_query, shard_update_data[shard])
                self._conn.commit()
            except Exception as e:
                if "databse is locked" in str(e):
                    time.sleep(10)
                self._conn.rollback()
                print(f"Error updating shard {shard_id}: {e}")
                raise e
        print(f"Took {time.time()-start_time:.2f} sec to update the database.")

    def process_dset(self):

        start_time = time.time()
        alldata = np.concatenate((self.data['position'][:], self.data['direction'][:]), axis = 1)
        all_pos_rot = rotate_wcte(alldata[:, :3])
        all_dir_rot = rotate_wcte(alldata[:, 3:])
        alldata_rot = np.concatenate((all_pos_rot, all_dir_rot), axis = 1)

        #wcsim output converts mm to cm
        all_bin_pos = self.rows[:, 1:4]/10.
        all_bin_dir = np.array([np.cos(self.rows[:, 4] * np.pi / 180.) * np.sin(self.rows[:, 5] * np.pi / 180.),
                                np.sin(self.rows[:, 4] * np.pi / 180.) * np.sin(self.rows[:, 5] * np.pi / 180.),
                                np.cos(self.rows[:, 5] * np.pi / 180.)])

        all_bin_dist = np.linalg.norm(alldata_rot[:, np.newaxis,:3] - all_bin_pos[np.newaxis, :,:], axis = 2)
        all_ang_diff = np.degrees(np.arccos(np.dot(all_dir_rot, all_bin_dir))) # use degrees so it's numerically closer to the position in cm

        dist_bin = np.sqrt(all_bin_dist**2 + all_ang_diff**2)
        nearest_grid_indices = np.argmin(dist_bin, axis = 1)

        hit_pmt = self.data['digi_pmt'][:]
        hit_charge = self.data['digi_charge'][:]
        hit_time = self.data['digi_time'][:]

        mask = (hit_pmt >= 0) & (hit_pmt < self.npmt)
        
        bindata = self.rows[:,1:6]
        sum_hit = np.zeros(shape=(len(self.rows), self.npmt), dtype=float)
        sum_charge = np.zeros(shape=(len(self.rows), self.npmt), dtype=float)
        sum_time = np.zeros(shape=(len(self.rows), self.npmt), dtype=float)

        bin_Qmax = np.zeros(shape=(len(self.rows),), dtype=float)

        sum_hit[nearest_grid_indices] = np.sum(hit_pmt[mask], axis = 0)
        sum_charge[nearest_grid_indices] = np.sum(hit_charge[mask], axis = 0)
        sum_time[nearest_grid_indices] = np.sum(hit_time[mask], axis = 0)

        bin_stats = np.bincount(nearest_grid_indices, minlength=len(self.rows))
        bin_Qmax[bin_stats>0] = np.max(sum_charge, axis=1)[bin_stats>0]/bin_stats[bin_stats>0]
        #print(np.sum(bin_stats>0), np.max(bin_stats), np.mean(bin_stats[bin_stats>0]), np.std(bin_stats[bin_stats>0]))

        sum_hit[bin_stats>0] /= bin_stats[bin_stats>0][:, np.newaxis]
        sum_charge[bin_stats>0] /= bin_stats[bin_stats>0][:, np.newaxis]
        sum_time[bin_stats>0] /= bin_stats[bin_stats>0][:, np.newaxis]

        self.dset['vertices'] = self.fh5.create_dataset('vertices', data=bindata.copy(),
                                                        compression=self.cmprs, compression_opts=self.cmprs_opt)
        if self.drop_unhit:
            nhits_total = np.sum(sum_hit>0, axis = 1)
            _, hit_indices = np.where(sum_hit > 0)
            assert len(hit_indices) == np.sum(nhits_total), 'The number of hits is not consistent.'
            concat_charge = sum_charge[sum_hit>0].flatten()
            concat_time = sum_time[sum_hit>0].flatten()
            assert len(concat_charge) == len(concat_time) and len(concat_charge) == len(hit_indices), 'The number of charges and times is not consistent.'

            self.dset['nhit_per_event'] = self.fh5.create_dataset('nhit_per_event', data=nhits_total.copy(),
                                                                  compression=self.cmprs, compression_opts=self.cmprs_opt)
            self.dset['hit_pmt']        = self.fh5.create_dataset('hit_pmt', data=hit_indices.copy(),
                                                                  compression=self.cmprs, compression_opts=self.cmprs_opt)
            self.dset['hit_charge']     = self.fh5.create_dataset('hit_charge', data=concat_charge.copy(),
                                                                  compression=self.cmprs, compression_opts=self.cmprs_opt)
            self.dset['hit_time']       = self.fh5.create_dataset('hit_time', data=concat_time.copy(),
                                                                  compression=self.cmprs, compression_opts=self.cmprs_opt)

        else:
            self.dset['hit_pmt']    = self.fh5.create_dataset('hit_pmt', data=sum_hit.copy(),
                                                              compression=self.cmprs, compression_opts=self.cmprs_opt)
            self.dset['hit_charge'] = self.fh5.create_dataset('hit_charge', data=sum_charge.copy(),
                                                              compression=self.cmprs, compression_opts=self.cmprs_opt)
            self.dset['hit_time']   = self.fh5.create_dataset('hit_time', data=sum_time.copy(),
                                                              compression=self.cmprs, compression_opts=self.cmprs_opt)
        self.fh5.close()
        print(f"Took {time.time()-start_time:.2f} sec to write to h5 file {self.outfile}.")

        return bin_stats, bin_Qmax

    def visualize_bin_selection(self, pts, dirs):
        import plotly.graph_objects as go
        import numpy as np
        fig = go.Figure()

        x_coords = [ self.r0_vox*np.cos(self.phi0_vox*np.pi/180.), self.r1_vox*np.cos(self.phi0_vox*np.pi/180.),
                     self.r1_vox*np.cos(self.phi1_vox*np.pi/180.), self.r0_vox*np.cos(self.phi1_vox*np.pi/180.),
                     self.r0_vox*np.cos(self.phi0_vox*np.pi/180.), self.r0_vox*np.cos(self.phi0_vox*np.pi/180.),                                          
                     self.r1_vox*np.cos(self.phi0_vox*np.pi/180.), self.r1_vox*np.cos(self.phi0_vox*np.pi/180.), 
                     self.r1_vox*np.cos(self.phi0_vox*np.pi/180.), self.r1_vox*np.cos(self.phi1_vox*np.pi/180.),
                     self.r1_vox*np.cos(self.phi1_vox*np.pi/180.), self.r1_vox*np.cos(self.phi1_vox*np.pi/180.),                     
                     self.r1_vox*np.cos(self.phi1_vox*np.pi/180.), self.r1_vox*np.cos(self.phi1_vox*np.pi/180.),
                     self.r0_vox*np.cos(self.phi1_vox*np.pi/180.), self.r0_vox*np.cos(self.phi1_vox*np.pi/180.),
                     self.r0_vox*np.cos(self.phi1_vox*np.pi/180.), self.r0_vox*np.cos(self.phi0_vox*np.pi/180.),                     
        ]
        
        y_coords = [ self.r0_vox*np.sin(self.phi0_vox*np.pi/180.), self.r1_vox*np.sin(self.phi0_vox*np.pi/180.),
                     self.r1_vox*np.sin(self.phi1_vox*np.pi/180.), self.r0_vox*np.sin(self.phi1_vox*np.pi/180.),
                     self.r0_vox*np.sin(self.phi0_vox*np.pi/180.), self.r0_vox*np.sin(self.phi0_vox*np.pi/180.),                                          
                     self.r1_vox*np.sin(self.phi0_vox*np.pi/180.), self.r1_vox*np.sin(self.phi0_vox*np.pi/180.), 
                     self.r1_vox*np.sin(self.phi0_vox*np.pi/180.), self.r1_vox*np.sin(self.phi1_vox*np.pi/180.),
                     self.r1_vox*np.sin(self.phi1_vox*np.pi/180.), self.r1_vox*np.sin(self.phi1_vox*np.pi/180.),                     
                     self.r1_vox*np.sin(self.phi1_vox*np.pi/180.), self.r0_vox*np.sin(self.phi1_vox*np.pi/180.),
                     self.r0_vox*np.sin(self.phi1_vox*np.pi/180.), self.r0_vox*np.sin(self.phi1_vox*np.pi/180.),
                     self.r0_vox*np.sin(self.phi1_vox*np.pi/180.), self.r0_vox*np.sin(self.phi0_vox*np.pi/180.),                     
        ]
        
        z_coords = [ self.z0_vox, self.z0_vox,
                     self.z0_vox, self.z0_vox,
                     self.z0_vox, self.z1_vox,
                     self.z1_vox, self.z0_vox,
                     self.z1_vox, self.z1_vox,                     
                     self.z1_vox, self.z0_vox,                     
                     self.z0_vox, self.z1_vox,                     
                     self.z1_vox, self.z0_vox,
                     self.z1_vox, self.z1_vox,
        ]        

        rows = self.rows[self.rows[:,0] % len(dirs) == 0]
        points = pts[ np.where(np.logical_and(pts[:, 2] >= np.min(rows[:, 3]), pts[:, 2] <= np.max(rows[:, 3]))) ]
        
        trape_data = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines',
            line=dict(color="rgba(108, 122, 137, 0.7)", width=3),
            #fillcolor="rgba(232, 232, 232, 0.5)",
        )

        trape_bins = go.Scatter3d(
            x=points[:,0],
            y=points[:,1],
            z=points[:,2],
            mode='markers',
            marker=dict(size=2, opacity=1, color='blue'),
        )

        points_bins = go.Scatter3d(
            x=rows[:,1],
            y=rows[:,2],
            z=rows[:,3],
            mode='markers',
            marker=dict(size=5, opacity=1, color='red')
        )

        fig.add_trace(trape_bins)
        fig.add_trace(trape_data)
        fig.add_trace(points_bins)

        return fig

def rotate_wcte(points: np.ndarray):
    """                             
    Rotates the given 3D points 90 degrees counterclockwise around the x-axis to correct the coordinates because of the WCTE geometry.
    """

    rotation_matrix = np.array([[1., 0,  0],
                                [0,  0, -1.],
                                [0,  1., 0]
                                ])
    rotated_points = np.dot(rotation_matrix, points.T).T
    return rotated_points
