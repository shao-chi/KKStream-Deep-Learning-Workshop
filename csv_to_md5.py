import hashlib
import csv
import numpy as np

f = open('best_submission.csv','r', newline='')
md = open('md5.csv','w', newline='')
rows = csv.reader(f)
writer = csv.writer(md)
for row in rows:
    ro = []
    for r in row:
        ro.append(hashlib.md5(r.encode()).hexdigest())
    writer.writerow(ro)

f.close()
md.close()