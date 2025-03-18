import sys, math
import pandas as pd

class Describer():
	def __init__(self, arg: str):
		try:
			self.data = pd.read_csv(arg)
		except (FileNotFoundError, Exception) as e:
			print("Error:", str(e))
			exit(1)

	def parser(self):
		self.num_features = []
		for col in self.data.columns:
			try:
				self.data[col].astype(float)
				self.num_features.append(col)
			except ValueError:
				continue

		self.p_data = []
		for col in self.num_features:
			tmp = [val for val in self.data[col] if not pd.isna(val)]
			if tmp:
				self.p_data.append(sorted(tmp))

	def count(self):
		count = []
		for data in self.p_data:
			count.append(len(data))
		return count

	def mean(self):
		mean = []
		for data in self.p_data:
			mean.append(sum(data) / len(data))
		return mean

	def std(self):
		std = []
		for data in self.p_data:
			mean = sum(data) / len(data)
			tmp = 0
			for val in data:
				tmp += (val - mean)**2
			std.append(math.sqrt(tmp / (len(data) - 1)))
		return std

	def min(self):
		min = []
		for data in self.p_data:
			min.append(data[0])
		return min

	def max(self):
		max = []
		for data in self.p_data:
			max.append(data[len(data) - 1])
		return max

	def percentile(self, percent):
		percentile = []
		for data in self.p_data:
			pos = (len(data) - 1) * percent
			npos = int(pos)
			dec = pos - npos
			if dec == 0:
				percentile.append(data[npos])
			else:
				percentile.append(data[npos] + (data[npos + 1] - data[npos]) * dec)
		return percentile

	def describe(self):
		self.parser()
		description = pd.DataFrame({
			"Count": self.count(),
			"Mean": self.mean(),
			"Std" : self.std(),
			"Min" : self.min(),
			"25%" : self.percentile(0.25),
			"50%" : self.percentile(0.50),
			"75%" : self.percentile(0.75),
			"Max" : self.max()
		}, index=self.num_features).T

		print(description)


def main():
	if len(sys.argv) != 2:
		print("Usage: python3 describe.py <dataset>")
		exit(1)
	d = Describer(sys.argv[1])
	d.describe()


if __name__ == "__main__":
	main()
