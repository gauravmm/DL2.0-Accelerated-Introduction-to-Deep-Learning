#! python3

# Here's how you do a local import in Python 3:
from . import model

# You can just import whichever dataset you want directly:
# Supported datasets are in the data package.
from data import test_dataset

print("Training model: {}".format(model.NAME))
print(test_dataset.get_train().__repr__())