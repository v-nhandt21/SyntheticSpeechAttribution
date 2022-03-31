import glob
import torchaudio
target_sr = 16000
A = []
import matplotlib.pyplot as plt
import numpy as np
for line in glob.glob("/home/nhandt23/Desktop/SPCUP/Data/part1/*"):

     wav, sr = torchaudio.load(line)
     A.append(wav.size(1)/16000)

print(min(A))
print(max(A))

plt.hist(A, density=False, bins=40)  # density=False would make counts
plt.ylabel('Number of Audio')
plt.xlabel('Second');
plt.savefig("duration.png")