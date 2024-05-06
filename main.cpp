#include <iostream>
#include <iomanip>
#include "layers.h"
#include "lossFunctions.h"
#include "optimizers.h"

typedef vector<vector<vector<vector<double>>>> v4;

int main() {
	vector<vector<double>> in = { {0.2, 0.9, -0.1}, {-0.2, 0, 0.1}, {0, 0.9, 0.5} };
	vector<vector<double>> out = { {1, 0, 0}, {0, 0, 1}, {1, 0, 0} };

	LinearLayer layer(3, 3, 3);
	Softmax softmax(3, 3);
	MSELoss loss(3, 3);
	Adam2d weights(3, 3);
	weights.l2 = 0.5;
	Adam1d biases(3);

	for (int e = 0; e < 20000; e++) {
		layer.forward(in);
		softmax.forward(layer.output);
		loss.calculate(softmax.output, out);
		if (e % 100 == 0) {
			for (int b = 0; b < 3; b++) {
				for (int i = 0; i < 3; i++) {
					cout << setw(10) << softmax.output[b][i];
				}
				cout << endl;
			}
			cout << loss.value << "\n\n";
		}
		softmax.backward(layer.output, loss.diff);
		layer.backward(in, softmax.diff);
		weights.step(layer.weights, layer.weightsDiff);
		biases.step(layer.biases, layer.biasesDiff);
		layer.zeroGradients();
	}

	return 0;
}