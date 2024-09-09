import numpy as np
import h5py
import sqlite3
import yaml
import glob
import os
import math

from datetime import datetime

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
            self.r0_vox = float(cfg_gen['r0_Vox'])
            self.r1_vox = float(cfg_gen['r1_Vox'])
            self.z0_vox = float(cfg_gen['z0_Vox'])
            self.z1_vox = float(cfg_gen['z1_Vox'])
            self.phi0_vox = float(cfg_gen['phi0_Vox'])
            self.phi1_vox = float(cfg_gen['phi1_Vox'])

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
                self.data[key] = hf[key][:]

    def create_grid(self):
        vox, pts = voxels(-self.zmax, self.zmax, 0, self.rmax, self.gap_space, self.n_phi_start)
        dirs = directions(self.gap_angle)
        #vox: 0:2 r-range, 2:4 phi-range, 4:6 z-range
        return vox, pts, dirs

    def create_db(self, pts, dirs):
        if os.path.exists(self.dbfile):
            return
        conn = sqlite3.connect(self.dbfile)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS bins
                     (id INTEGER, x REAL, y REAL, z REAL, phi REAL, theta REAL, stats INTEGER, Qmax REAL)''')
        for i, p in enumerate(pts):
            for j, d in enumerate(dirs):
                c.execute("INSERT INTO bins VALUES (?,?,?,?,?,?,?,?)", (i*len(dirs)+j, p[0], p[1], p[2], d[1], d[0], 0, 0))
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
        conn.create_function("SQRT",  1, math.sqrt)
        conn.create_function("POW",   2, math.pow)
        conn.create_function("ABS",   1, math.fabs)
        conn.create_function("ATAN2", 2, math.atan2)
        c = conn.cursor()

        query = '''
        SELECT id, x, y, z, phi, theta, stats, Qmax
        FROM bins
        WHERE ((SQRT(POW(x,2)+POW(y,2)) BETWEEN ? AND ?)
        AND (x/SQRT(POW(x,2)+POW(y,2)) BETWEEN ? AND ?)
        AND (y/SQRT(POW(x,2)+POW(y,2)) BETWEEN ? AND ?)
        AND (z BETWEEN ? AND ?))
        ;
        '''
        #OR (
        #(ABS(SQRT(POW(x,2)+POW(y,2)) - ?) < ? OR ABS(SQRT(POW(x,2)+POW(y,2)) - ?) < ?)
        #AND 
        #((ABS((ATAN2(x, y)*180./? + 360.) % 360. - ?) < ?) OR (ABS((ATAN2(x, y)*180./? + 360.) % 360. - ?) < ?))
        #AND
        #(ABS(z - ?) < ? OR ABS(z - ?) < ?)
        #self.r0_vox, 0.5*self.gap_space, self.r1_vox, 0.5*self.gap_space,
        #math.pi, self.phi0_vox, 0.5*math.fabs(self.phi1_vox-self.phi0_vox), math.pi, self.phi1_vox, 0.5*math.fabs(self.phi1_vox-self.phi0_vox),
        #self.z0_vox, 0.5*self.gap_space, self.z1_vox, 0.5*self.gap_space

        c.execute(query, (self.r0_vox, self.r1_vox,
                          min(math.cos(self.phi0_vox*math.pi/180.), math.cos(self.phi1_vox*math.pi/180.)), max(math.cos(self.phi0_vox*math.pi/180.), math.cos(self.phi1_vox*math.pi/180.)),
                          min(math.sin(self.phi0_vox*math.pi/180.), math.sin(self.phi1_vox*math.pi/180.)), max(math.sin(self.phi0_vox*math.pi/180.), math.sin(self.phi1_vox*math.pi/180.)),
                          self.z0_vox, self.z1_vox))
        self.rows = np.array(c.fetchall())
        print(f"Fetched {len(self.rows)} bins that covers the generated voxel from db")
        conn.close()

    def process_dset(self):

        alldata = np.concatenate((self.data['position'][:]*10., self.data['direction'][:]), axis = 1)
        hit_pmt = self.data['digi_pmt'][:]    
        hit_charge = self.data['digi_charge'][:]
        hit_time = self.data['digi_time'][:]

        mask = (hit_pmt >= 0) & (hit_pmt < self.npmt)
        
        bindata = np.zeros(shape=(len(self.rows), 5), dtype=float)
        sum_hit = np.zeros(shape=(len(self.rows), self.npmt), dtype=float)
        sum_charge = np.zeros(shape=(len(self.rows), self.npmt), dtype=float)
        sum_time = np.zeros(shape=(len(self.rows), self.npmt), dtype=float)

        bin_stats = np.zeros(shape=(len(self.rows),), dtype=int)
        bin_Qmax = np.zeros(shape=(len(self.rows),), dtype=float)

        for i, d in enumerate(alldata):
            pos_rot = rotate_wcte(d[:3])
            dir_rot = rotate_wcte(d[3:])

            min_dist = self.gap_space
            min_angle = 1.
            min_idx = -1
            
            for j, r in enumerate(self.rows):
                bin_dir = np.array([np.cos(r[4]*np.pi/180.)*np.sin(r[5]*np.pi/180.), np.sin(r[4]*np.pi/180.)*np.sin(r[5]*np.pi/180.), np.cos(r[5]*np.pi/180.)])
                dist_angle = np.linalg.norm(dir_rot - bin_dir)
                if (np.linalg.norm(r[1:4] - pos_rot) <= min_dist) and (dist_angle <= min_angle):
                    min_dist = np.linalg.norm(r[1:4] - pos_rot)                    
                    min_angle = dist_angle
                    min_idx = j

            if min_idx == -1:
                continue
            else:
                bindata[min_idx] = self.rows[min_idx][1:6]
                sum_hit[min_idx][hit_pmt[i][mask[i]]] += 1
                sum_charge[min_idx][hit_pmt[i][mask[i]]] += hit_charge[i][mask[i]]
                sum_time[min_idx][hit_pmt[i][mask[i]]] += hit_time[i][mask[i]]
                bin_stats[min_idx] += 1
                bin_Qmax[min_idx] = max(bin_Qmax[min_idx], np.max(sum_charge[min_idx])/bin_stats[min_idx])


        conn = sqlite3.connect(self.dbfile)
        c = conn.cursor()
        sql_update_query = """UPDATE bins
                              SET stats = ?, Qmax = ?
                              WHERE id = ?"""
        for i, r in enumerate(self.rows):
            print(i, bin_stats[i], bin_Qmax[i])
            c.execute(sql_update_query, (bin_stats[i]+r[-2], max(bin_Qmax[i],r[-1]), r[0]))
        conn.commit()
        conn.close()

        sum_hit[bin_stats>0] /= bin_stats[bin_stats>0][:, np.newaxis]
        sum_charge[bin_stats>0] /= bin_stats[bin_stats>0][:, np.newaxis]
        sum_time[bin_stats>0] /= bin_stats[bin_stats>0][:, np.newaxis]


        self.dset['vertices']   = self.fh5.create_dataset('vertices', data=bindata.copy(),
                                                          compression=self.cmprs, compression_opts=self.cmprs_opt)
        self.dset['hit_pmt']    = self.fh5.create_dataset('hit_pmt', data=sum_hit.copy(),
                                                          compression=self.cmprs, compression_opts=self.cmprs_opt)
        self.dset['hit_charge'] = self.fh5.create_dataset('hit_charge', data=sum_charge.copy(),
                                                          compression=self.cmprs, compression_opts=self.cmprs_opt)
        self.dset['hit_time']   = self.fh5.create_dataset('hit_time', data=sum_time.copy(),
                                                          compression=self.cmprs, compression_opts=self.cmprs_opt)
        self.fh5.close()
        print(f"Written to h5 file {self.outfile}.")

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
    rotated_points = np.dot(rotation_matrix, points)
    return rotated_points
