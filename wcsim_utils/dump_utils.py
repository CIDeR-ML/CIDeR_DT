"""
Python 3 script for processing a list of ROOT files into .h5 files
Transcribed from event_dump.py and np_to_digihit/truehit_array_hdf5.py

Authors: Junjie Xia
"""
import numpy as np
import awkward as ak
import h5py
import glob
import ROOT
import os

from datetime import datetime
from cachetools import cached, LRUCache
from .root_utils import WCSim
#from memory_profiler import profile
#from line_profiler import profile

ROOT.gROOT.SetBatch(True)
cache = LRUCache(maxsize=100)

dtype_map = {
    'np.int32': np.int32,
    'np.float32': np.float32,
    'np.float64': np.float64,
    'np.int64': np.int64,
    'np.int16': np.int16,
    'np.uint8': np.uint8,
}

class WCSimRead(WCSim):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.file_name = cfg['data']['file_name']
        self.outfile = cfg['data']['output_file']
        self.nevents_per_file = cfg['data']['nevents_per_file']
        self.npmts = cfg['detector']['npmts']

        self.cmprs = cfg['format'].get('compression', 'gzip')
        self.cmprs_opt = cfg['format'].get('compression_opt', 5)

        self.write_event_info = 'event_info' in cfg['data']['root_branches']
        self.write_hit_photons = 'hit_photons' in cfg['data']['root_branches']
        self.write_digi_hits = 'digi_hits' in cfg['data']['root_branches']
        self.write_tracks = 'tracks' in cfg['data']['root_branches']
        self.write_trigger = 'trigger' in cfg['data']['root_branches']

        self.infiles = glob.glob(self.file_name)
        self.wcsim = [] # list of WCSim file names
        self.chain = None
        self.nevents = 0
        self.root_inputs = {}
        self.dset = {}
        self.chain = None

        self.track_filled = False
        self.trigger_filled = False
        
        self.read_root_files()
        self.initialized = False

    def read_root_files(self):
        self.chain = ROOT.TChain("wcsimT")        
        for infile in self.infiles:
            try:
                if not os.path.exists(infile):
                    raise FileNotFoundError(f"{infile} not found. Skipping.")
                self.chain.Add(infile)
                self.wcsim.append(infile)
                print("Added", infile, "to chain")
            except FileNotFoundError as e:
                print(e)
        super().__init__(self.chain)
    
    def get_nevents(self):
        return self.nevents

    @cached(cache)
    def get_digitized_hits(self, ev):
        for value in self.cfg['data']['root_branches']['digi_hits']:
            self.root_inputs['digi_' + value[0]].begin_list()
        self.get_event(ev)
        for t in range(self.ntrigger):
            self.get_trigger(t)
            for hit in self.trigger.GetCherenkovDigiHits():
                pmt_id = hit.GetTubeId() - 1

                self.root_inputs['digi_pmt'][ev].integer(pmt_id)
                self.root_inputs['digi_charge'][ev].real(hit.GetQ())
                self.root_inputs['digi_time'][ev].real(hit.GetT())
                self.root_inputs['digi_trigger'][ev].integer(t)

        for value in self.cfg['data']['root_branches']['digi_hits']:
            self.root_inputs['digi_' + value[0]].end_list()

    @cached(cache)
    def get_hit_photons(self, ev):
        self.get_event(ev)
        for value in self.cfg['data']['root_branches']['hit_photons']:
            if value[0] != 'PE':
                self.root_inputs['true_'+value[0]].begin_list()
        for t in range(self.ntrigger):
            self.get_trigger(t)
            for h in self.trigger.GetCherenkovHits():
                self.root_inputs["true_PE"][ev][h.GetTubeID()-1] = h.GetTotalPe(1)
            # need to do the following because GetChrenkovHitTimes takes a long time
            photons = self.trigger.GetCherenkovHitTimes()
            for it in range(photons.GetEntries()):
                p = photons[it]
                self.root_inputs['true_end_time'].real(p.GetTruetime())
                self.root_inputs['true_track'].integer(p.GetParentID())
                try:  # Only works with new tracking branch of WCSim
                    self.root_inputs['true_start_time'].real(p.GetPhotonStartTime())
                    self.root_inputs['true_start_position'].append([p.GetPhotonStartPos(i) / 10 for i in range(3)])
                    self.root_inputs['true_end_position'].append([p.GetPhotonEndPos(i) / 10 for i in range(3)])
                except AttributeError:  # leave as zeros if not using tracking branch
                    pass
        for value in self.cfg['data']['root_branches']['hit_photons']:
            if value[0] != 'PE':
                self.root_inputs['true_'+value[0]].end_list()

    @cached(cache)
    def get_tracks(self, ev):
        self.get_event(ev)
        for value in self.cfg['data']['root_branches']['tracks']:
            self.root_inputs['track_'+value[0]].begin_list()
        for t in range(self.ntrigger):
            self.get_trigger(t)
            for track in self.trigger.GetTracks():                
                self.root_inputs['track_id'].integer(track.GetId())
                self.root_inputs['track_pid'].integer(track.GetIpnu())
                self.root_inputs['track_start_time'].real(track.GetTime())
                self.root_inputs['track_energy'].real(track.GetE())
                self.root_inputs['track_start_position'].append([track.GetStart(i) for i in range(3)])
                self.root_inputs['track_stop_position'].append([track.GetStop(i) for i in range(3)])
                self.root_inputs['track_parent'].integer(track.GetParenttype())
                self.root_inputs['track_flag'].integer(track.GetFlag())
                self.track_filled = True
        for value in self.cfg['data']['root_branches']['tracks']:
            self.root_inputs['track_'+value[0]].end_list()

    @cached(cache)
    def get_triggers(self, ev):
        self.get_event(ev)
        for value in self.cfg['data']['root_branches']['trigger']:
            self.root_inputs['trigger_'+value[0]].begin_list()
        for t in range(self.ntrigger):
            self.get_trigger(t)
            self.root_inputs['trigger_time'].real(self.trigger.GetHeader().GetDate())
            trig_type = self.trigger.GetTriggerType()
            if trig_type > np.iinfo(np.int32).max:
                trig_type = -1
            self.root_inputs['trigger_type'].integer(trig_type)
            self.trigger_filled = True
        for value in self.cfg['data']['root_branches']['trigger']:
            self.root_inputs['trigger_'+value[0]].end_list()

    def initialize_array(self):
        self.root_inputs["event_ids"]   = np.empty(self.nevents, dtype=np.int32)
        self.root_inputs["root_file"]  = np.empty(self.nevents, dtype=object)

        if self.write_event_info:
            for value in self.cfg['data']['root_branches']['event_info']:
                self.root_inputs[value[0]] = np.empty((self.nevents, int(value[-1])), dtype=dtype_map.get(value[1]))

        if self.write_hit_photons:
            for value in self.cfg['data']['root_branches']['hit_photons']:
                if value[0] == "PE":
                    self.root_inputs['true_'+value[0]] = np.empty((self.nevents, int(value[-1])), dtype=dtype_map.get(value[1]))
                else:
                    self.root_inputs['true_'+value[0]]  = ak.ArrayBuilder()

        if self.write_digi_hits:
            for value in self.cfg['data']['root_branches']['digi_hits']:
                #self.root_inputs['digi_'+value[0]]  = np.empty((self.nevents, int(value[-1])), dtype=dtype_map.get(value[1]))
                self.root_inputs['digi_'+value[0]] = ak.ArrayBuilder()

        if self.write_tracks:
            for value in self.cfg['data']['root_branches']['tracks']:
                self.root_inputs['track_'+value[0]] = ak.ArrayBuilder()

        if self.write_trigger:
            for value in self.cfg['data']['root_branches']['trigger']:
                self.root_inputs['trigger_'+value[0]] = ak.ArrayBuilder()

    def create_h5(self):
        print("ouput file:", self.outfile)
        if os.path.exists(self.outfile):
            os.remove(self.outfile)
        self.fh5 = h5py.File(self.outfile, 'w')
        self.fh5.attrs['timestamp'] = str(datetime.now())

        self.dset['PATHS']      = self.fh5.create_dataset("root_files", shape=(self.nevents,),dtype=h5py.special_dtype(vlen=str), compression=self.cmprs, compression_opts=self.cmprs_opt)
        self.dset['event_ids']  = self.fh5.create_dataset("event_ids", shape=(self.nevents, ), dtype=np.int16, compression=self.cmprs, compression_opts=self.cmprs_opt)
        
        if self.write_event_info:
            for value in self.cfg['data']['root_branches']['event_info']:
                self.dset[value[0]]            = self.fh5.create_dataset(value[0], shape=(self.nevents, int(value[-1])), dtype=dtype_map.get(value[1]), compression=self.cmprs, compression_opts=self.cmprs_opt)

        #if self.write_digi_hits:
        #    for value in self.cfg['data']['root_branches']['digi_hits']:
        #        self.dset['digi_'+value[0]]    = self.fh5.create_dataset('digi_'+value[0], shape=(self.nevents, int(value[-1])), dtype=dtype_map.get(value[1]), compression=self.cmprs, compression_opts=self.cmprs_opt)

    def dump_array(self):
        if not self.initialized:
            self.initialize_array()
            self.create_h5()
            self.initilized = True

        for ev in range(self.nevents):
            
            if self.write_event_info:
                self.event_info = self.get_event_info()
                for value in self.cfg['data']['root_branches']['event_info']:
                    self.root_inputs[value[0]][ev] = self.event_info[value[0]]

            if self.write_hit_photons:
                self.get_hit_photons(ev)

            if self.write_digi_hits:
                self.get_digitized_hits(ev)

            if self.write_tracks:
                self.get_tracks(ev)

            if self.write_trigger:
                self.get_triggers(ev)

            self.root_inputs['event_ids'][ev] = ev
            self.root_inputs['root_file'][ev] = self.wcsim[int(ev/self.nevents_per_file)]

    #@profile
    def dump_to_h5(self):
        # labels -> pid
        # veto   -> trigger type
        # veto2  -> trigger time
        self.dset['PATHS'][:]       = self.root_inputs['root_file']
        self.dset['event_ids'][:]    = self.root_inputs['event_ids']

        if self.write_event_info:
            for value in self.cfg['data']['root_branches']['event_info']:
                self.dset[value[0]][:] = self.root_inputs[value[0]]

        if self.write_hit_photons:
            for value in self.cfg['data']['root_branches']['hit_photons']:
                if value[0] == 'PE':
                    self.dset['true_PE'] = self.fh5.create_dataset('true_PE', data = self.root_inputs['true_PE'])
                else:
                    arr_fill = self.root_inputs['true_'+value[0]].snapshot()
                    padded = ak.pad_none(arr_fill, target=ak.max(ak.num(arr_fill, axis=1)), axis=1)
                    filled = ak.fill_none(padded, -999)
                    np_filled = ak.to_numpy(filled)
                    self.dset['true_'+value[0]] = self.fh5.create_dataset('true_'+value[0], data = np_filled.copy())

        if self.write_digi_hits:
            for value in self.cfg['data']['root_branches']['digi_hits']:
                arr_fill = self.root_inputs['digi_' + value[0]].snapshot()
                padded = ak.pad_none(arr_fill, target=ak.max(ak.num(arr_fill, axis=1)), axis=1)
                filled = ak.fill_none(padded, -999)
                np_filled = ak.to_numpy(filled)
                self.dset['digi_' + value[0]] = self.fh5.create_dataset('digi_' + value[0], data=np_filled.copy())

        if self.write_tracks and self.track_filled:
            for value in self.cfg['data']['root_branches']['tracks']:
                arr_fill = self.root_inputs['track_'+value[0]].snapshot()
                padded = ak.pad_none(arr_fill, target=ak.max(ak.num(arr_fill, axis=1)), axis=1)
                filled = ak.fill_none(padded, -999)
                np_filled = ak.to_numpy(filled)                
                self.dset['track_'+value[0]] = self.fh5.create_dataset('track_'+value[0], data = np_filled.copy())
                    

        if self.write_trigger and self.trigger_filled:
            for value in self.cfg['data']['root_branches']['trigger']:
                arr_fill = self.root_inputs['trigger_'+value[0]].snapshot()
                padded = ak.pad_none(arr_fill, target=ak.max(ak.num(arr_fill, axis=1)), axis=1)
                filled = ak.fill_none(padded, -999)
                np_filled = ak.to_numpy(filled)
                self.dset['trigger_'+value[0]] = self.fh5.create_dataset('trigger_'+value[0], data = np_filled.copy())
                    
                
        self.fh5.close()
        print("Written to h5 file!") 
