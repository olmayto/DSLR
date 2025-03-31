import sys, json
import pandas as pd
import numpy as np

class Trainer:
	def __init__(self, dataset, lr=0.01, epochs=1000):
		try:
			data = pd.read_csv(dataset).fillna(0)
		except (FileNotFoundError, Exception) as e:
			print("Error:", str(e))
			exit(1)

		self.X = data.select_dtypes(include="number").drop(
			columns=["Index", "Astronomy", "Care of Magical Creatures", "Potions"]
			).reset_index(drop=True)
		self.X = (self.X - self.X.mean()) / self.X.std()

		houses_map = {"Gryffindor": 0, "Hufflepuff": 1, "Ravenclaw": 2, "Slytherin": 3}
		self.y = data["Hogwarts House"].map(houses_map).reset_index(drop=True)

		self.lr = lr
		self.epochs = epochs
		self.classifiers = {}

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def train_binary(self, y):
		m, n = self.X.shape
		weights = np.zeros(n)
		bias = 0

		for _ in range(self.epochs):
			linear_pred = np.dot(self.X, weights) + bias
			predictions = self.sigmoid(linear_pred)

			dw = (1 / m) * np.dot(self.X.T, (predictions - y))
			db = (1 / m) * np.sum(predictions - y)

			weights = weights - self.lr * dw
			bias = bias - self.lr * db
		
		return weights, bias

	def train(self):
		classes = np.unique(self.y)

		for c in classes:
			bin_y = np.where(self.y == c, 1 , 0)
			weights, bias = self.train_binary(y=bin_y)
			self.classifiers[c] = {"weights": weights.tolist(), "bias": float(bias)}

		self.classifiers_to_json()

	def classifiers_to_json(self):
		try:
			with open("model.json", "w") as f:
				json.dump({int(k): v for k, v in self.classifiers.items()}, f)
		except Exception as e:
			print("Error:", str(e))
			exit(1)


def main():
	if len(sys.argv) != 2:
		print("Usage: python3 logreg_train.py <dataset>")
		exit(1)
	d = Trainer(sys.argv[1])
	d.train()


if __name__ == "__main__":
	main()
