#include <iostream>
#include "layers.h"
#include "lossFunctions.h"
#include "optimizers.h"

int main() {
	ConvolutionalLayer layer(4, 4, 1, 1, 1);
	Flatten31 flatten(1, 4, 4, 1);
	MSELoss loss(16, 1);
	Adam4d weights(1, 1, 3, 3);
	Adam1d biases(1);
	vector<vector<vector<vector<double>>>> in = { {
		{
			{1, 0.2, 0.5, -0.2},
			{-0.2, 0.3, 0.1, 0.5},
			{0.1, 0.8, 0.5, -0.2},
			{0, -0.2, 0.1, 0.2}
		}
	} };
	vector<vector<double>> out = { {1, 0.2, 0.5, -0.2, -0.2, 0.3, 0.1, 0.5,
		0.1, 0.8, 0.5, -0.2, 0, -0.2, 0.1, 0.2} };
	for (int i = 0; i < out[0].size(); i++) {
		out[0][i] = out[0][i] * (2) + 0.2;
	}

	for (int e = 0; e < 10000; e++) {
		layer.forward(in);
		flatten.forward(layer.output);
		loss.calculate(flatten.output, out);
		if (e % 100 == 0) {
			for (int i = 0; i < flatten.output[0].size(); i++) {
				cout << flatten.output[0][i] << " ";
			}
			cout << endl << loss.value << "\n\n\n";
		}
		flatten.backward(layer.output, loss.diff);
		layer.backward(in, flatten.diff);
		weights.step(layer.weights, layer.weightsDiff);
		biases.step(layer.biases, layer.biasesDiff);
		layer.zeroGradients();
	}

	return 0;
}