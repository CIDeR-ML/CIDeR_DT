import numpy as np
import h5py
import sqlite3
import pandas as pd
import yaml
import glob
import os

from wcprod.utils import voxels, directions
from wcsim_utils.root_utils import rotate_wcte

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
            self.r0_vox = float(cfg_gen['r0_vox'])
            self.r1_vox = float(cfg_gen['r1_vox'])
            self.z0_vox = float(cfg_gen['z0_vox'])
            self.z1_vox = float(cfg_gen['z1_vox'])
            self.phi0_vox = float(cfg_gen['phi0_vox'])
            self.phi1_vox = float(cfg_gen['phi1_vox'])

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

        self.wall_cut = cfg['Action']['wall_cut']
        self.towall_cut = cfg['Action']['towall_cut']

        self.npmt = cfg['Detector']['npmt']

        self.dset = {}
        self.cmprs = cfg['Format'].get('compression', 'gzip')
        self.cmprs_opt = cfg['Format'].get('compression_opts', 5)

    def load_data(self):
        # Load the data from the h5 file
        with h5py.File(self.infile, 'r') as hf:
            for key in self.dset_names:
                self.data[key] = hf['key'][:]

    def create_grid(self):
        vox, pts = voxels(-self.zmax, self.zmax, 0, self.rmax, self.gap_space, self.n_phi_start)
        dirs = directions(self.gap_angle, self.n_phi_start)
        #vox: 0:2 r-range, 2:4 phi-range, 4:6 z-range
        return vox, pts, dirs

    def create_db(self, pts, dirs):
        conn = sqlite3.connect(self.dbfile)
        c = conn.cursor()
        c.execute('''CREATE TABLE bins IF NOT EXISTS bins
                     (id INTEGER, x REAL, y REAL, z REAL, phi REAL, theta REAL, stats INTEGER, Qmax REAL)''',)
        for i, p in enumerate(pts):
            for j, d in enumerate(dirs):
                c.execute("INSERT INTO bins VALUES (?,?,?,?,?,?,?,?)", (i*len(dirs)+j, p[0], p[1], p[2], d[0], d[1], 0, 0))
        conn.commit()
        conn.close()

    def create_final_h5(self):
        print("ouput file:", self.outfile)
        if os.path.exists(self.outfile):
            os.remove(self.outfile)
        self.fh5 = h5py.File(self.outfile, 'w')
        self.fh5.attrs['timestamp'] = str(datetime.now())

    def select_bins_from_db(self):
        conn = sqlite3.connect(self.dbfile)
        c = conn.cursor()
        query = '''
        SELECT id, x, y, z, phi, theta, stats, Qmax
        FROM bins
        WHERE ((x**2+y**2)**0.5 > ? AND (x**2+y**2)**0.5 < ?
        AND np.acos(x/(x**2+y**2)**0.5)*180./np.pi > ? AND np.acos(x/(x**2+y**2)**0.5)*180./np.pi < ?
        AND z > ? AND z < ?)
        OR ( 
        (np.abs((x**2+y**2)**0.5 - ?) < 0.5*self.gap_space OR np.abs((x**2+y**2)**0.5 - ?) < 0.5*self.gap_space)
        AND 
        (np.abs(np.acos(x/(x**2+y**2)**0.5)*180./np.pi - ?) < 0.5*self.gap_angle OR np.abs(np.acos(x/(x**2+y**2)**0.5)*180./np.pi - ?) < 0.5*self.gap_angle)
        AND
        (np.abs(z - ?) < 0.5*self.gap_space OR np.abs(z - ?) < 0.5*self.gap_space)
        );
        '''
        c.execute(query, (self.r0_vox, self.r1_vox, self.phi0_vox, self.phi1_vox, self.z0_vox, self.z1_vox,
                               self.r0_vox, self.r1_vox, self.phi0_vox, self.phi1_vox, self.z0_vox, self.z1_vox))
        self.rows = np.array(c.fetchall())
        conn.close()

    def process_dset(self):

        alldata = np.column_stack(self.data['position'][:], self.data['direction'][:])
        hit_pmt = self.data['digi_pmt'][:]
        hit_charge = self.data['digi_charge'][:]
        hit_time = self.data['digi_time'][:])

        bindata = np.zeros(shape=(len(self.rows), 5), dtype=float)
        sum_pmt = np.zeros(shape=(len(self.rows), self.npmt), dtype=float)
        sum_charge = np.zeros(shape=(len(self.rows), self.npmt), dtype=float)
        sum_time = np.zeros(shape=(len(self.rows), self.npmt), dtype=float)

        bin_stats = np.zeros(shape=(len(self.rows), 1), dtype=int)
        bin_Qmax = np.zeros(shape=(len(self.rows), 1), dtype=float)

        min_dist = 999
        min_angle = 999
        min_idx = -1

        for i, d in enumerate(alldata):
            pos_rot = rotate_wcte(d[:3])
            dir_rot = rotate_wcte(d[3:])
            dir_angle = np.array[np.arccos(dir_rot[0]/(dir_rot[0]**2+dir_rot[1]**2)**0.5, np.arccos(dir_rot[2])]

            for j, r in enumerate(self.rows):
                if np.linalg.norm(r[1:4] - pos_rot) < min_dist and np.linalg.norm(r[4:6] - dir_angle) < min_angle:
                    min_dist = np.linalg.norm(r[1:4] - pos_rot)
                    min_angle = np.linalg.norm(r[4:6] - dir_angle)
                    min_idx = j

            if min_idx == -1:
                continue
            else:
                bindata[min_idx, 0:5] = self.rows[min_idx][1:6]
                sum_hit[min_idx][hit_pmt[i]] += 1
                sum_charge[min_idx][hit_pmt[i]] += hit_charge[i]
                sum_time[min_idx][hit_pmt[i]] += hit_time[i]

                bin_stats[min_idx] += 1
                bin_Qmax[min_idx] = max(bin_Qmax[min_idx], max(hit_charge[i]))


        conn = sqlite3.connect(self.dbfile)
        c = conn.cursor()
        sql_update_query = """UPDATE bins
                              SET stats = ?, Qmax = ?
                              WHERE idx = ?"""
        for r in enumerate(self.rows):
            c.execute(sql_update_query, (bin_stats[i], bin_Qmax[i], r[0]))
        conn.commit()
        conn.close()

        sum_time /= sum_hit


        self.dset['vertices']   = self.fh5.create_dataset('vertices', data=bindata.copy(),
                                                          compression=self.cmprs, compression_opts=self.cmprs_opt)
        self.dset['hit_pmt']    = self.fh5.create_dataset('hit_pmt', data=sum_pmt.copy(),
                                                          compression=self.cmprs, compression_opts=self.cmprs_opt)
        self.dset['hit_charge'] = self.fh5.create_dataset('hit_charge', data=sum_charge.copy(),
                                                          compression=self.cmprs, compression_opts=self.cmprs_opt)
        self.dset['hit_time']   = self.fh5.create_dataset('hit_time', data=sum_time.copy(),
                                                          compression=self.cmprs, compression_opts=self.cmprs_opt)
        self.fh5.close()
        print(f"Written to h5 file {self.outfile}.")
