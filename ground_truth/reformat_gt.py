
import sys
import os

SEQUENCES_BASE_PATH = '../sequence'

sorted_sequences = sorted(os.listdir(
    SEQUENCES_BASE_PATH), key=lambda x: int(os.path.splitext(x)[0]))

f = open("gt.txt", "r")
for line in f:
  id, pid, x1, y1, w, h = line.split(',')[0:6]
  line = "person %s %s %s %s\n" % (x1, y1, str(float(x1) + float(w)), str(float(y1) + float(h)))
  file_name = 'output_gt/' + sorted_sequences[int(id) - 1].split('.')[0] + '.txt'
  open(file_name, 'a+').write(line)