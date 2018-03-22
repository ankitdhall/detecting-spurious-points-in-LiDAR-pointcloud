import random as random
random.seed(41)

from os import listdir
from os.path import isfile, join


mypath = "data/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

onlyfiles = sorted(onlyfiles)
random.shuffle(onlyfiles)

train = onlyfiles[:160]
test = onlyfiles[160:]

text_file = open("train.txt", "w")
for filename in train:
	text_file.write("%s\n" % filename)
text_file.close()

text_file = open("test.txt", "w")
for filename in test:
	text_file.write("%s\n" % filename)
text_file.close()

print "train:", len(train), "test:", len(test)

