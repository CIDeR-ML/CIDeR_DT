import yaml
import fire

from rebin_utils import wc_binning

def main(config_file, config_gen):
    cfg = dict()
    cfg_gen = dict()
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    with open(config_gen, 'r') as f:
        cfg_gen = yaml.safe_load(f)

    _rebin = wc_binning(cfg, cfg_gen)
    _rebin.load_data()
    vox, pts, dirs = _rebin.create_grid()
    _rebin.create_db(pts, dirs)
    _rebin.create_final_h5()
    _rebin.select_bins_from_db()
    _rebin.process_dset()

if __name__ == '__main__':
    fire.Fire(main)