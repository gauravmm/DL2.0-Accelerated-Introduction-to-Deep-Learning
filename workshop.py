#! python3

import argparse
import importlib
import logging
import pathlib

import data

# Logging
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('[%(asctime)s %(levelname)-3s @%(name)s] %(message)s', datefmt='%H:%M:%S'))
logging.basicConfig(level=logging.DEBUG, handlers=[console])
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logger = logging.getLogger("workshop")



def run(args):
    print("""
 ______   _____       _____       ____   
|_   _ `.|_   _|     / ___ `.   .'    '. 
  | | `. \ | |      |_/___) |  |  .--.  |
  | |  | | | |   _   .'____.'  | |    | |
 _| |_.' /_| |__/ | / /_____  _|  `--'  |
|______.'|________| |_______|(_)'.____.' 
                                         
""")

    mod_name = "{}.{}".format(args.example, args.split)
    logger.info("Running script at {}".format(mod_name))

    try:
        importlib.import_module(mod_name)
    except Exception as e:
        logger.exception(e)
        logger.error("Uhoh, the script halted with an error.")

def path(d):
    try:
        return importlib.util.find_spec(d).name
    except Exception as e:
        raise argparse.ArgumentTypeError("Example {} cannot be located.".format(d))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run examples from the DL 2.0 Workshop.')
    parser.add_argument('example', type=path, help='the folder name of the example you want to run')
    parser.add_argument('split', choices=['train', 'test'], help='train the example or evaluate it')

    run(parser.parse_args())
