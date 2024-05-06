#include <iostream>
#include <iomanip>
#include "layers.h"
#include "lossFunctions.h"
#include "optimizers.h"

void print(vector<double>& v) {
	for (int i = 0; i < v.size(); i++) {
		cout << setw(6) << v[i];
	}
	cout << endl;
}

int main() {
	cout.precision(3);
	int batches = 10, sizeIn = 20, sizeOut = 10;
	vector<vector<double>> in(batches, vector<double>(sizeIn));
	vector<vector<double>> out(batches, vector<double>(sizeIn));
	for (int b = 0; b < batches; b++) {
		for (int i = 0; i < sizeIn; i++) {
			in[b][i] = (double)rand() / (double)RAND_MAX;
			cout << setw(6) << in[b][i];
		}
		cout << endl;
	}
	cout << endl;

	for (int b = 0; b < batches; b++) {
		for (int i = 0; i < sizeOut; i++) {
			out[b][i] = (double)rand() / (double)RAND_MAX;
			cout << setw(6) << out[b][i];
		}
		cout << endl;
	}
	cout << endl;

	LinearLayer layer1(sizeIn, 15, batches);
	Sigmoid1d activation(15, batches);
	Dropout1d drop1(15, batches, 0.5);
	LinearLayer layer2(15, sizeOut, batches);
	MSELoss loss(sizeOut, batches);

	Adam2d weights1(sizeIn, 15);
	Adam1d biases1(15);
	Adam2d weights2(15, sizeOut);
	Adam1d biases2(sizeOut);

	for (int e = 0; e < 20000; e++) {
		layer1.forward(in);
		activation.forward(layer1.output);
		drop1.forward(activation.output);
		layer2.forward(drop1.output);
		loss.calculate(layer2.output, out);

		cout << setw(16) << loss.value;

		layer2.backward(drop1.output, loss.diff);
		drop1.backward(activation.output, layer2.diff);
		activation.backward(layer1.output, drop1.diff);
		layer1.backward(in, activation.diff);

		weights1.step(layer1.weights, layer1.weightsDiff);
		biases1.step(layer1.biases, layer1.biasesDiff);
		weights2.step(layer2.weights, layer2.weightsDiff);
		biases2.step(layer2.biases, layer2.biasesDiff);
		layer1.zeroGradients();
		layer2.zeroGradients();

		layer1.forward(in);
		activation.forward(layer1.output);
		drop1.forward(activation.output, false);
		layer2.forward(drop1.output);
		loss.calculate(layer2.output, out);
		cout << setw(6) << loss.value << endl;
	}


	return 0;
}