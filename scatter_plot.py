import pandas as pd
import matplotlib.pyplot as plt

try:
	raw_data = pd.read_csv("datasets/dataset_train.csv")
except (FileNotFoundError, Exception) as e:
	print("Error:", str(e))
	exit(1)

data = raw_data.select_dtypes(include="number").drop(columns="Index")
corr = data.corr().unstack().sort_values(ascending=False, key=abs).reset_index().query("level_0 != level_1")
corr["pairs"] = corr.apply(lambda row: tuple(sorted([row["level_0"], row["level_1"]])), axis=1)
corr = corr.drop_duplicates(subset="pairs")

pairs = corr["pairs"].to_list()
data.insert(0, 'Hogwarts House', raw_data['Hogwarts House'])
houses = data["Hogwarts House"].sort_values().unique()
colors = {"Gryffindor": "red", "Slytherin": "green", "Ravenclaw": "blue", "Hufflepuff": "yellow"}

current_index = [0]
fig, ax = plt.subplots(figsize=(10, 6))
fig.canvas.manager.set_window_title("scatter_plot.py")

def plot_scatter(index):
	ax.clear()
	pair = pairs[index]
	p_corr = float(corr[corr["pairs"] == pair][0].iloc[0])

	for house in houses:
		house_data1 = data[data["Hogwarts House"] == house][pair[0]]
		house_data2 = data[data["Hogwarts House"] == house][pair[1]]
		ax.scatter(house_data1, house_data2, label=house, color=colors.get(house), edgecolors="grey")

	ax.set_title(f"Score correlation between {pair[0]} and {pair[1]}")
	ax.set_xlabel(pair[0])
	ax.set_ylabel(pair[1])
	ax.legend(loc="upper right")
	ax.text(0.015, 0.975, f"Correlation: {p_corr:.4f}", transform=ax.transAxes, fontsize=12, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
	plt.draw()

def on_key(event):
	if event.key == "right":
		current_index[0] = (current_index[0] + 1) % len(pairs)
	elif event.key == "left":
		current_index[0] = (current_index[0] - 1) % len(pairs)
	plot_scatter(current_index[0])

plot_scatter(current_index[0])
fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
