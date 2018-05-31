import random as random
random.seed(41)

from os import listdir
from os.path import isfile, join


def split_all():
	mypath = "data/"
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

	onlyfiles = sorted(onlyfiles)
	random.shuffle(onlyfiles)

	TRAIN_FRACTION = 0.8
	TOTAL_DATA = len(onlyfiles)

	TRAIN_NUM = int(TOTAL_DATA*TRAIN_FRACTION)

	train = onlyfiles[:TRAIN_NUM]
	test = onlyfiles[TRAIN_NUM:]

	text_file = open("train.txt", "w")
	for filename in train:
		text_file.write("%s\n" % filename)
	text_file.close()

	text_file = open("test.txt", "w")
	for filename in test:
		text_file.write("%s\n" % filename)
	text_file.close()

	print "train:", len(train), "test:", len(test)

def split_scenes(test_ids):
	train, test = [], []
	
	mypath = "data/"
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

	# onlyfiles = sorted(onlyfiles)
	# random.shuffle(onlyfiles)

	for filename in onlyfiles:

		in_test = False
		for test_id in test_ids:
			if test_id in filename:
				test.append(filename)
				in_test = True
				break
		if not in_test:
			train.append(filename)


	text_file = open("train.txt", "w")
	for filename in train:
		text_file.write("%s\n" % filename)
	text_file.close()

	text_file = open("test.txt", "w")
	for filename in test:
		text_file.write("%s\n" % filename)
	text_file.close()

	print "train:", len(train), "test:", len(test)

split_scenes(test_ids = ["7_", "8_", "9_", "10_"])
