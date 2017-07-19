#! python3

# Here's how you do a local import in Python 3:
from . import model

# You can just import whichever dataset you want directly:
# Supported datasets are in the data package.
from data import yt8m, utilities

print("Training model: {}".format(model.NAME))

# Get .tfrecord files
for s in yt8m.get_train():
    print(s)
