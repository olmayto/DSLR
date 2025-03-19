import pandas as pd
import matplotlib.pyplot as plt

try:
	raw_data = pd.read_csv("datasets/dataset_train.csv")
except (FileNotFoundError, Exception) as e:
	print("Error:", str(e))
	exit(1)

data = raw_data.select_dtypes(include="number").drop(columns="Index")
data.insert(0, 'Hogwarts House', raw_data['Hogwarts House'])
long_data = data.melt(id_vars=["Hogwarts House"], var_name="Course", value_name="Score")
distribution = long_data.groupby(["Course", "Hogwarts House"])["Score"].mean().reset_index()
distribution_std = distribution.groupby(["Course"])["Score"].std().sort_values().reset_index()

courses = distribution_std["Course"].to_list()
houses = distribution["Hogwarts House"].unique()
colors = {"Gryffindor": "red", "Slytherin": "green", "Ravenclaw": "blue", "Hufflepuff": "yellow"}

current_index = [0]
fig, ax = plt.subplots(figsize=(10, 6))
fig.canvas.manager.set_window_title("histogram.py")

def plot_histogram(index):
	ax.clear()
	course = courses[index]
	std = distribution_std[distribution_std["Course"] == course].iloc[0, 1]

	for house in houses:
		house_data = data[data["Hogwarts House"] == house][course]
		ax.hist(house_data, bins=10, alpha=0.5, label=house, color=colors.get(house), edgecolor="grey")

	ax.set_title(f"Score distribution for {course}")
	ax.set_xlabel("Score")
	ax.set_ylabel("Frequency")
	ax.legend(loc="upper right")
	ax.text(0.015, 0.975, f"STD: {std:.3f}", transform=ax.transAxes, fontsize=12, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
	plt.draw()

def on_key(event):
	if event.key == "right":
		current_index[0] = (current_index[0] + 1) % len(courses)
	elif event.key == "left":
		current_index[0] = (current_index[0] - 1) % len(courses)
	plot_histogram(current_index[0])

plot_histogram(current_index[0])
fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
