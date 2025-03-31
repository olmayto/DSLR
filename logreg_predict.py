import sys, json
import pandas as pd
import numpy as np

class Predictor:
	def __init__(self, dataset, model):
		try:
			data = pd.read_csv(dataset).fillna(0)
			with open(model, "r") as f:
				self.classifiers = json.load(f)
		except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
			print("Error:", str(e))
			exit(1)

		self.X = data.select_dtypes(include="number").drop(
			columns=["Hogwarts House", "Index", "Astronomy", "Care of Magical Creatures", "Potions"]
			).reset_index(drop=True)
		self.X = (self.X - self.X.mean()) / self.X.std()

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def predict(self):
		predictions = {}
		for c, classifier in self.classifiers.items():
			weights = np.array(classifier["weights"])
			bias = classifier["bias"]
			linear_pred = np.dot(self.X, weights) + bias
			predictions[c] = self.sigmoid(linear_pred)

		result = np.array([max(predictions, key=lambda k: predictions[k][i]) for i in range(self.X.shape[0])])
		houses_map = {0: "Gryffindor", 1: "Hufflepuff", 2: "Ravenclaw", 3: "Slytherin"}
		houses = [houses_map[int(val)] for val in result]
		output = pd.DataFrame({"Hogwarts House": houses})
		output.index.name = "Index"
		output.to_csv("houses.csv")


def main():
	if len(sys.argv) != 3:
		print("Usage: python3 logreg_predict.py <dataset> <model>")
		exit(1)
	p = Predictor(sys.argv[1], sys.argv[2])
	p.predict()


if __name__ == "__main__":
	main()
