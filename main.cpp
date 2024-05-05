#include <iostream>
#include "layers.h"
#include "lossFunctions.h"
#include "optimizers.h"

int main() {
	MaxPooling pool(1, 4, 4, 1);
	ConvolutionalLayer conv(4, 4, 1, 1, 1);
	Adam4d weights(1, 1, 3, 3);
	weights.l2 = 0.01;
	Adam1d biases(1);
	Flatten31 flatten(1, 2, 2, 1);
	MSELoss loss(4, 1);
	vector<vector<vector<vector<double>>>> in = {
		{
			{
				{0.1, 0.6, -0.8, 0},
				{0.4, -0.6, 0.5, 0},
				{-0.6, 0.8, 0.8, 0},
				{0.9, 0.6, 0, 1}
			}
		}
	};
	vector<vector<double>> out = {
		{1.3, 0.5, -1.8, 0.2}
	};

	for (int e = 0; e < 50000; e++) {
		conv.forward(in);
		pool.forward(conv.output);
		flatten.forward(pool.output);
		loss.calculate(flatten.output, out);

		for (int r = 0; r < conv.output[0][0].size(); r++) {
			for (int c = 0; c < conv.output[0][0][r].size(); c++) {
				cout << conv.output[0][0][r][c] << " ";
			}
			cout << endl;
		}
		cout << loss.value << endl << endl;

		flatten.backward(pool.output, loss.diff);
		pool.backward(conv.output, flatten.diff);

		/*for (int r = 0; r < pool.output[0][0].size(); r++) {
			for (int c = 0; c < pool.output[0][0][r].size(); c++) {
				cout << pool.memory[0][0][r][c] << " ";
			}
			cout << endl;
		}
		cout << endl;
		for (int r = 0; r < pool.diff[0][0].size(); r++) {
			for (int c = 0; c < pool.diff[0][0][r].size(); c++) {
				cout << pool.diff[0][0][r][c] << " ";
			}
			cout << endl;
		}*/
		cout << "\n\n\n";

		conv.backward(in, pool.diff);
		weights.step(conv.weights, conv.weightsDiff);
		biases.step(conv.biases, conv.biasesDiff);
	}

	return 0;
}