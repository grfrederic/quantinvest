import pandas as pd
import sys
import matplotlib.pyplot as plt


# data source
if len(sys.argv) < 2:
    print(f"Usage: ./{sys.argv[0]} data.csv")
    exit(1)
else:
    dataFile = sys.argv[1]


# load data
data = pd.read_csv(dataFile)


# plot
print(data.columns)

tm = data["time"]
ap = data["AP"]

#plt.plot(data["time"], data[['AP', 'ARR', 'ARW', 'G', 'OP', 'ORR', 'ORW']])
plt.plot(data["time"], data[['G']])


# legend
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.50, 2.20), loc="upper right", borderaxespad=0.)


# done
plt.show()
