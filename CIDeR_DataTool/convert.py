import yaml
import fire

from wcsim_utils import WCSimRead

def main(config_file):
    cfg = dict()
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    wcsim = WCSimRead(cfg)
    wcsim.read_root_files()
    wcsim.dump_array()
    wcsim.dump_to_h5()

if __name__ == '__main__':
    fire.Fire(main)
