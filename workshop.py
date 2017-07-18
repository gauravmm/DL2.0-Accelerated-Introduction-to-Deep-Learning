#! python3

import argparse
import importlib
import logging
import os

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

    has_effect = False

    if args.datasets:
        download_data()
        has_effect = True

    if args.pretrained:
        if args.example:
            download_pretrained(example)
        else:
            for example in ["cnn", "gan"]:
                download_pretrained(example)
        has_effect = True

    if args.example and args.split:
        try:
            mod_name = "{}.{}".format(args.example, args.split)
            logger.info("Running script at {}".format(mod_name))

            mod = importlib.import_module(mod_name)
        except Exception as e:
            logger.exception(e)
            logger.error("Uhoh, the script halted with an error.")
    else:
        if not has_effect:
            logger.error("Script halted without any effect. To run code, use command:\npython3 workshop.py <example name> {train, test}")

def download_data():
    from data import nodules, cifar10, yt8m

    for mod in [nodules, cifar10, yt8m]:
        print("Downloading: {}".format(mod.__name__))
        mod.get_train()
        mod.get_test()

def download_pretrained(example):
    pass


def path(d):
    try:
        assert os.path.isdir(d)
        return d
    except Exception as e:
        raise argparse.ArgumentTypeError("Example {} cannot be located.".format(d))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run examples from the DL 2.0 Workshop.')
    parser.add_argument('--datasets', action='store_true', help='download all datasets')
    parser.add_argument('--pretrained', action='store_true', help='download pretrained weights for an example if specified, or all examples otherwise')
    parser.add_argument('example', nargs="?", type=path, help='the folder name of the example you want to run')
    parser.add_argument('split', nargs="?", choices=['train', 'test'], help='train the example or evaluate it')

    run(parser.parse_args())
