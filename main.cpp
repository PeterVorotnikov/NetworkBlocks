#include <iostream>
#include <iomanip>
#include "layers.h"
#include "lossFunctions.h"
#include "optimizers.h"

typedef vector<vector<vector<vector<double>>>> v4;

int main() {
	vector<vector<double>> in = { {0.2, 0.9, -0.1}, {-0.2, 0, 0.1}, {0, 0.9, 0.5} };
	vector<int> out = { 0, 1, 2 };

	LinearLayer layer(3, 3, 3);
	Adam2d weights(3, 3);
	Adam1d biases(3);
	CategoricalCrossentropyLoss loss(3, 3);

	for (int e = 0; e < 30000; e++) {
		layer.forward(in);
		loss.calculate(layer.output, out);
		cout << loss.value << endl;
		layer.backward(in, loss.diff);
		weights.step(layer.weights, layer.weightsDiff);
		biases.step(layer.biases, layer.biasesDiff);
		layer.zeroGradients();
	}

	return 0;
}