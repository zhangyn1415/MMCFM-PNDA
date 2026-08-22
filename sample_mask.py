import sys

from sample import main

if __name__ == "__main__":
    args = ["--task", "mask", "--config_path", "configs/PanNuke_Mask.yaml", "--data_type", "PanNuke"] + sys.argv[1:]
    main(args)
