#! python3

# Here's how you do a local import in Python 3:
from . import model

# You can just import whichever dataset you want directly:
# Supported datasets are in the data package.
from data import cifar10

print("Training model: {}".format(model.NAME))
