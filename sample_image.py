import sys

from sample import main

if __name__ == "__main__":
    args = ["--task", "image", "--config_path", "configs/PanNuke.yaml"] + sys.argv[1:]
    main(args)
