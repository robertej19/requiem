import pandas as pd
import matplotlib as plt

df = pd.read_pickle("skim8_005032_filtered.pkl")

ax = df.plot.hist("Q2")
fig = ax.get_figure()
plt.plot.show()
#fig.savefig("test.jpg")