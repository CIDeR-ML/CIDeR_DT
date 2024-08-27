"""
Python 3 script for processing a list of ROOT files into .h5 files
Transcribed from event_dump.py and np_to_digihit/truehit_array_hdf5.py

Authors: Junjie Xia
"""
import numpy as np
import h5py
import glob
from cachetools import cached, LRUCache

from root_utils import WCSim, WCSimFile

ROOT.gROOT.SetBatch(True)
cache = LRUCache(maxsize=100)
class WCSimRead():
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.event_info_cfg = cfg['event_info']

        self.file_name = cfg['data']['file_name']
        self.outfile = cfg['data']['output_file']
        self.npmts = cfg['detector']['npmts']

        self.write_event_info = 'event_info' in cfg['data']['root_branches']
        self.write_hit_photons = 'hit_photons' in cfg['data']['root_branches']
        self.write_digi_hits = 'digi_hits' in cfg['data']['root_branches']
        self.write_tracks = 'tracks' in cfg['data']['root_branches']
        self.write_trigger = 'trigger' in cfg['data']['root_branches']

        self.infiles = glob.glob(self.file_name)
        self.wcsim = []
        self.nevents = 0
        self.root_inputs = {}
        self.dset = {}

        self.read_root_files()
        self.initialized = False

    def read_root_files(self):
        for infile in self.infiles:
            self.wcsim.append(WCSimFile(infile))
            self.nevents += self.wcsim[-1].nevent()
    def get_nevent_perfile(self, i):
        return self.wcsim[i].nevent()
    def get_nevents(self):
        return self.nevents
    @cached(cache)
    def get_event(self, i):
        file_index = int(i/len(self.wcsim))
        event_index = i%len(self.wcsim)
        self.wcsim[file_index].get_event(event_index)
        return file_index, event_index

    @cached(cache)
    def get_digitized_hits(self, ev, fidx):
        for t in range(self.wcsim[fidx].ntrigger):
            self.wcsim[fidx].get_trigger(t)
            for hit in self.wcsim[fidx].trigger.GetCherenkovDigiHits():
                pmt_id = hit.GetTubeId() - 1

                self.root_inputs["digi_pmt"][ev][pmt_id] = 1
                self.root_inputs["digi_charge"][ev][pmt_id] = hit.GetQ()
                self.root_inputs["digi_time"][ev][pmt_id] = hit.GetT()
                self.root_inputs["digi_trigger"][ev][pmt_id] = t

    @cached(cache)
    def get_hit_photons(self, ev, fidx):
        for t in range(self.wcsim[fidx].ntrigger):
            self.wcsim[fidx].get_trigger(t)
            for h in self.wcsim[fidx].trigger.GetCherenkovHitTimes():
                self.root_inputs["true_PE"][ev][h.GetTubeID()-1] = h.GetTotalPe(1)
            # need to do the following because GetChrenkovHitTimes takes a long time
            photons = self.trigger.GetCherenkovHitTimes()
            for it in range(photons.GetEntries()):
                p = photons[it]
                self.root_inputs['true_end_time'][ev][it] = p.GetTruetime()
                self.root_inputs['true_track'][ev][it] = p.GetParentID()
                try:  # Only works with new tracking branch of WCSim
                    self.root_inputs['true_start_time'][ev][it] = p.GetPhotonStartTime()
                    self.root_inputs['true_start_position'][ev][:] = [p.GetPhotonStartPos(i) / 10 for i in range(3)]
                    self.root_inputs['true_end_position'][ev][:] = [p.GetPhotonEndPos(i) / 10 for i in range(3)]
                except AttributeError:  # leave as zeros if not using tracking branch
                    pass

    @cached(cache)
    def get_tracks(self, ev, fidx):
        for t in range(self.wcsim[fidx].ntrigger):
            self.wcsim[fidx].get_trigger(t)
            for track in self.trigger.GetTracks():
                self.root_inputs['track_id'][ev][-1] = track.GetId()
                self.root_inputs['track_pid'][ev][-1] = track.GetIpnu()
                self.root_inputs['track_start_time'][ev][-1] = track.GetTime()
                self.root_inputs['track_energy'][ev][-1] = track.GetE()
                self.root_inputs['track_start_position'][ev][:] = [track.GetStart(i) for i in range(3)]
                self.root_inputs['track_stop_position'][ev][:] = [track.GetStop(i) for i in range(3)]
                self.root_inputs['track_parent'][ev][-1]  = track.GetParenttype()
                self.root_inputs['track_flag'][ev][-1] = track.GetFlag()

    @cached(cache)
    def get_triggers(self, ev, fidx):
        for t in range(self.wcsim[fidx].ntrigger):
            self.wcsim[fidx].get_trigger(t)
            self.root_inputs['trigger_time'][t] = self.wcsim[fidx].trigger.GetHeader().GetDate()
            trig_type = self.wcsim[fidx].trigger.GetTriggerType()
            if trig_type > np.iinfo(np.int32).max:
                trig_type = -1
            self.root_inputs['trigger_type'][t] = trig_type

    def initialize_array(self):
        self.root_inputs["event_id"]   = np.empty(self.nevents, dtype=np.int32)
        self.root_inputs["root_file"]  = np.empty(self.nevents, dtype=object)

        if self.write_event_info:
            for value in self.cfg['data']['root_branches']['event_info']:
                self.root_inputs[value[0]] = np.empty((self.nevents, int(value[-1])), dtype=value[1])

        if self.write_hit_photons:
            for value in self.cfg['data']['root_branches']['hit_photons']:
                self.root_inputs['true_'+value[0]]  = np.empty((self.nevents, int(value[-1])), dtype=value[1])

        if self.write_digi_hits:
            for value in self.cfg['data']['root_branches']['digi_hits']:
                self.root_inputs['digi_'+value[0]]  = np.empty((self.nevents, int(value[-1])), dtype=value[1])

        if self.write_tracks:
            for value in self.cfg['data']['root_branches']['tracks']:
                self.root_inputs['track_'+value[0]] = np.empty((self.nevents, int(values[-1])), dtype=value[1])

        if self.write_trigger:
            for value in self.cfg['data']['root_branches']['trigger']:
                self.root_inputs['trigger_'+value[0]] = np.empty((self.nevents, int(values[-1])), dtype=value[1])

    def create_h5(self):
        print("ouput file:", self.outfile)
        self.fh5 = h5py.File(self.outfile, 'w')
        self.fh5.attrs['timestamp'] = str(datetime.now())

        #self.dset['labels']     = self.fh5.create_dataset("labels", shape=(self.nevents,) dtype=np.int32)
        self.dset['PATHS']      = self.fh5.create_dataset("root_files", shape=(self.nevents, 1),dtype=h5py.special_dtype(vlen=str))
        self.dset['event_ids']  = self.fh5.create_dataset("event_ids", shape=(self.nevents, 1), dtype=np.int32)
        if self.write_event_info:
            for value in self.cfg['data']['root_branches']['event_info']:
                self.dset[value[0]]            = self.fh5.create_dataset(value[0], shape=(self.nevents, int(value[-1])), dtype=value[1])

        if write_hit_photons:
            for value in self.cfg['data']['root_branches']['hit_photons']:
                self.dset['true_'+value[0]]    = self.fh5.create_dataset('true_'+value[0], shape=(self.nevents, int(value[-1])), dtype=value[1])

        if write_digi_hits:
            for value in self.cfg['data']['root_branches']['digi_hits']:
                self.dset['digi_'+value[0]]    = self.fh5.create_dataset('digi_'+value[0], shape=(self.nevents, int(value[-1])), dtype=value[1])

        if write_tracks:
            for value in self.cfg['data']['root_branches']['tracks']:
                self.dset['track_'+value[0]]   = self.fh5.create_dataset('track_'+value[0], shape=(self.nevents, int(value[-1])), dtype=value[1])

        if write_trigger:
            for value in self.cfg['data']['root_branches']['trigger']:
                self.dset['trigger_'+value[0]] = self.fh5.create_dataset('trigger_'+value[0], shape=(self.nevents, int(value[-1])), dtype=value[1])

        self.dset['veto']       = self.fh5.create_dataset("veto",shape=(self.nevents,),dtype=np.bool_)
        self.dset['veto2']      = self.fh5.create_dataset("veto2",shape=(self.nevents,),dtype=np.bool_)

    def dump_array(self):
        if not self.initialized:
            self.initialize_array()
            self.create_h5()
            self.initilized = True

        for ev in range(self.nevents):
            fidx, eidx = self.get_event(ev)

            if self.write_event_info:
                self.event_info = self.wcsim[fidx].get_event_info()
                for value in self.cfg['data']['root_branches']['event_info']:
                    self.root_inputs[value[0]][ev] = self.event_info[value[0]]

            if self.write_hit_photons:
                self.get_hit_photons(ev, fidx)

            if self.write_digi_hits:
                self.get_digitized_hits(ev, fidx)

            if self.write_tracks:
                self.get_tracks(ev, fidx)

            if self.write_trigger:
                self.get_triggers(ev, fidx)

            self.root_inputs['event_id'][ev] = ev
            self.root_inputs['root_file'][ev] = self.wcsim[fidx]

    def dump_to_h5(self):
        # labels -> pid
        # veto   -> trigger type
        # veto2  -> trigger time
        self.dset['PATHS'][:]       = self.root_inputs['root_file']
        self.dset['event_id'][:]    = self.root_inputs['event_id']

        if self.write_event_info:
            for value in self.cfg['data']['root_branches']['event_info']:
                self.dset[value[0]][:] = self.root_inputs[value[0]]

        if self.write_hit_photons:
            for value in self.cfg['data']['root_branches']['hit_photons']:
                self.dset['true_'+value[0]][:] = self.root_inputs['true_'+value[0]]

        if self.write_digi_hits:
            for value in self.cfg['data']['root_branches']['digi_hits']:
                self.dset['digi_'+value[0]][:] = self.root_inputs['digi_'+value[0]]

        if self.write_tracks:
            for value in self.cfg['data']['root_branches']['tracks']:
                self.dset['track_'+value[0]][:] = self.root_inputs['track_'+value[0]]

        if self.write_trigger:
            for value in self.cfg['data']['root_branches']['trigger']:
                self.dset['trigger_'+value[0]][:] = self.root_inputs['trigger_'+value[0]]

        self.fh5.close()
