#! python3

# You can just import whichever dataset you want directly:
# Supported datasets are in the data package.
from data import nodules, cifar10, yt8m

for mod in [nodules, cifar10, yt8m]:
    print("Downloading: {}".format(mod.__name__))
    mod.get_train()
    mod.get_test()
