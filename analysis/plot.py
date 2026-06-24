import argparse

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("log_file", type=str)

nums = []
args = parser.parse_args()
with open(args.log_file) as f:
    print(f.readline())
    while line := f.readline().strip():
        num = float(line.split(" ")[-1])
        nums.append(num)

plt.plot(np.array(nums))
plt.show()
