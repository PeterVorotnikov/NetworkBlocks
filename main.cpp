#include <iostream>
#include "layers.h"
#include "lossFunctions.h"
#include "optimizers.h"

int main() {
	LinearLayer layer(3, 2, 2);
	Adam1d biases(2);
	Adam2d weights(3, 2);
	MSELoss loss(2, 1);
	vector<vector<double>> in = { {-0.5, 0.6, 0.1} };
	vector<vector<double>> out = { {0.7, 0.2} };
	for (int e = 0; e < 10000; e++) {
		layer.forward(in);
		loss.calculate(layer.output, out);
		for (int i = 0; i < layer.output.size(); i++) {
			for (int j = 0; j < layer.output[i].size(); j++) {
				cout << layer.output[i][j] << " ";
			}
			cout << endl;
		}
		cout << loss.value << endl << endl;
		layer.backward(in, loss.diff);
		biases.step(layer.biases, layer.biasesDiff);
		weights.step(layer.weights, layer.weightsDiff);
	}

	return 0;
}