#! python3

# Here's how you do a local import in Python 3:
from . import model

# You can just import whichever dataset you want directly:
# Supported datasets are in the data package.
from data import nodules, utilities

print("Training model: {}".format(model.NAME))

gen = utilities.infinite_generator(nodules.get_train(), 10)
for i in range(10):
    x, y = next(gen)
    print(x.shape, y.shape)
