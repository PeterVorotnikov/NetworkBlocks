#include <iostream>
#include "layers.h"
#include "lossFunctions.h"
#include "optimizers.h"

int main() {
	LinearLayer layer1(3, 3, 3);
	Sigmoid activation(3, 3);
	LinearLayer layer2(3, 2, 3);
	Adam1d biases1(3, 0.0003, 0.9, 0.99);
	Adam2d weights1(3, 3, 0.0003, 0.9, 0.99);
	Adam1d biases2(2, 0.0003, 0.9, 0.99);
	Adam2d weights2(3, 2, 0.0003, 0.9, 0.99);
	MSELoss loss(2, 3);
	vector<vector<double>> in = { {-0.5, 0.6, 0.1}, {0.8, 0.8, 0}, {0.1, 0.2, 0.6} };
	vector<vector<double>> out = { {0.7, 0.2}, {-0.7, 0.3}, {2, 1} };
	for (int e = 0; e < 10000; e++) {
		layer1.forward(in);
		activation.forward(layer1.output);
		layer2.forward(activation.output);
		loss.calculate(layer2.output, out);
		for (int b = 0; b < in.size(); b++) {
			for (int i = 0; i < out[b].size(); i++) {
				cout << layer2.output[b][i] << " ";
			}
			cout << endl;
		}
		cout << loss.value << "\n\n";
		layer2.backward(activation.output, loss.diff);
		activation.backward(layer1.output, layer2.diff);
		layer1.backward(in, activation.diff);
		biases1.step(layer1.biases, layer1.biasesDiff);
		weights1.step(layer1.weights, layer1.weightsDiff);
		biases2.step(layer2.biases, layer2.biasesDiff);
		weights2.step(layer2.weights, layer2.weightsDiff);
		layer1.zeroGradients();
		layer2.zeroGradients();
	}

	return 0;
}