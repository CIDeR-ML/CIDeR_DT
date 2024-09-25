import fire
from rebin_utils import wc_binning

def main(config_file, config_gen):
    _rebin = wc_binning(config_file, config_gen)
    vox, pts, dirs = _rebin.create_grid()
    _rebin.create_sharded_db(pts, dirs)

if __name__ == '__main__':
    fire.fire(main)