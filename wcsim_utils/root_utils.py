import ROOT
import os
import numpy as np

ROOT.gSystem.Load(os.environ['WCSIMDIR'] + "/lib/libWCSimRoot.so")

class WCSim:
    def __init__(self, tree):
        self.tree = tree
        self.nevents = self.tree.GetEntries()
        print("number of entries in the tree: " + str(self.nevents))
        # Get first event and trigger to prevent segfault when later deleting trigger to prevent memory leak
        self.tree.GetEvent(0)
        self.current_event = 0
        self.event = self.tree.wcsimrootevent
        self.ntrigger = self.event.GetNumberOfEvents()
        self.trigger = self.event.GetTrigger(0)
        self.current_trigger = 0

    def get_event(self, ev):
        # Delete previous triggers to prevent memory leak (only if file does not change)
        triggers = [self.event.GetTrigger(i) for i in range(self.ntrigger)]
        oldfile = self.tree.GetCurrentFile()
        self.tree.GetEvent(ev)
        if self.tree.GetCurrentFile() == oldfile:
            [t.Delete() for t in triggers]
        self.current_event = ev
        self.event = self.tree.wcsimrootevent
        self.ntrigger = self.event.GetNumberOfEvents()

    def get_trigger(self, trig):
        self.trigger = self.event.GetTrigger(trig)
        self.current_trigger = trig
        return self.trigger

    def get_first_trigger(self):
        first_trigger = 0
        first_trigger_time = 9999999.0
        for index in range(self.ntrigger):
            self.get_trigger(index)
            trigger_time = self.trigger.GetHeader().GetDate()
            if trigger_time < first_trigger_time:
                first_trigger_time = trigger_time
                first_trigger = index
        return self.get_trigger(first_trigger)

    def get_event_info(self):
        self.get_trigger(0)
        tracks = self.trigger.GetTracks()
        # Primary particles with no parent are the initial simulation
        particles = [t for t in tracks if t.GetFlag() == 0 and t.GetParenttype() == 0]
        # Check there is exactly one particle with no parent:
        if len(particles) == 1:
            # Only one primary, this is the particle being simulated
            return {
                "pid": particles[0].GetIpnu(),
                "position": [particles[0].GetStart(i) for i in range(3)],
                "direction": [particles[0].GetDir(i) for i in range(3)],
                "energy": particles[0].GetE()
            }
        # Particle with flag -1 is the incoming neutrino or 'dummy neutrino' used for gamma
        # WCSim saves the gamma details (except position) in the neutrino track with flag -1
        neutrino = [t for t in tracks if t.GetFlag() == -1]
        # Check for dummy neutrino that actually stores a gamma that converts to e+ / e-
        isConversion = len(particles) == 2 and {p.GetIpnu() for p in particles} == {11, -11}
        if isConversion and len(neutrino) == 1 and neutrino[0].GetIpnu() == 22:
            return {
                "pid": 22,
                "position": [particles[0].GetStart(i) for i in range(3)], # e+ / e- should have same position
                "direction": [neutrino[0].GetDir(i) for i in range(3)],
                "energy": neutrino[0].GetE()
            }
        # Check for dummy neutrino from old gamma simulations that didn't save the gamma info
        if isConversion and len(neutrino) == 1 and neutrino[0].GetIpnu() == 12 and neutrino[0].GetE() < 0.0001:
            # Should be a positron/electron pair from a gamma simulation (temporary hack since no gamma truth saved)
            momentum = [sum(p.GetDir(i) * p.GetP() for p in particles) for i in range(3)]
            norm = np.sqrt(sum(p ** 2 for p in momentum))
            return {
                "pid": 22,
                "position": [particles[0].GetStart(i) for i in range(3)],  # e+ / e- should have same position
                "direction": [p / norm for p in momentum],
                "energy": sum(p.GetE() for p in particles)
            }
        # Otherwise something else is going on... guess info from the primaries
        momentum = [sum(p.GetDir(i) * p.GetP() for p in particles) for i in range(3)]
        norm = np.sqrt(sum(p ** 2 for p in momentum))
        return {
            "pid": 0,  # there's more than one particle so just use pid 0
            "position": [sum(p.GetStart(i) for p in particles)/len(particles) for i in range(3)],  # average position
            "direction": [p / norm for p in momentum],  # direction of sum of momenta
            "energy": sum(p.GetE() for p in particles)  # sum of energies
        }


def get_label(infile):
    if "_gamma" in infile:
        label = 0
    elif "_e" in infile:
        label = 1
    elif "_mu" in infile:
        label = 2
    elif "_pi0" in infile:
        label = 3
    else:
        print("Unknown input file particle type")
        raise SystemExit
    return label
