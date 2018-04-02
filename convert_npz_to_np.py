import random as random
random.seed(41)

from os import listdir
from os.path import isfile, join

import numpy as np

annotation_dir = "data_npz/"
filename = [f for f in listdir(annotation_dir) if isfile(join(annotation_dir, f))]

# print filename

for i in range(len(filename)):

	loaded = np.load(annotation_dir + filename[i][:-4] + ".npz")
	decoded_file = loaded["gt"]
	decoded_file = decoded_file[()]

	# print decoded_file.keys()

	print(i)

	np.save(filename[i][:-4] + ".npy", np.array(decoded_file))