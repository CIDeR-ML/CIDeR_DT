import yaml
import fire

from rebin_utils import wc_binning

def main(config_file, config_gen):
    _rebin = wc_binning(config_file, config_gen)
    _rebin.load_data()
    vox, pts, dirs = _rebin.create_grid()
    _rebin.create_db(pts, dirs)
    _rebin.create_final_h5()
    _rebin.select_bins_from_db()
    _rebin.process_dset()

if __name__ == '__main__':
    fire.Fire(main)
