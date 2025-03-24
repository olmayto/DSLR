import pandas as pd
import matplotlib.pyplot as plt

try:
	raw_data = pd.read_csv("datasets/dataset_train.csv")
except (FileNotFoundError, Exception) as e:
	print("Error:", str(e))
	exit(1)

data = raw_data.select_dtypes(include="number").drop(columns="Index").dropna()
houses = raw_data["Hogwarts House"].loc[data.index]
colors_dict = {"Gryffindor": "red", "Slytherin": "green", "Ravenclaw": "blue", "Hufflepuff": "yellow"}
colors = houses.map(colors_dict)

scatter_matrix = pd.plotting.scatter_matrix(data, figsize=(12, 12), diagonal="hist", alpha=0.5, c=colors)

for i in range(data.shape[1]):
	ax = scatter_matrix[i, i]
	ax.clear()

	for house in colors_dict.keys():
		subset = data.loc[houses == house, data.columns[i]]
		ax.hist(subset, bins=30, color=colors_dict[house], alpha=0.6, label=house)

plt.gcf().canvas.manager.set_window_title("pair_plot.py")
plt.show()
