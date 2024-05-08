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
	cout.precision(5);
	int batches = 100, sizeIn = 200, sizeOut = 10;
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
	BatchNormalization1d norm(15, batches);
	LinearLayer layer2(15, sizeOut, batches);
	MSELoss loss(sizeOut, batches);

	Adam2d weights1(sizeIn, 15, 0.01);
	Adam1d biases1(15, 0.01);
	Adam2d weights2(15, sizeOut, 0.01);
	Adam1d biases2(sizeOut, 0.01);
	Adam1d gamma(15);
	Adam1d beta(15);

	for (int e = 0; e < 2000; e++) {
		layer1.forward(in);
		activation.forward(layer1.output);
		norm.forward(activation.output);
		layer2.forward(norm.output);
		loss.calculate(layer2.output, out);

		cout << setw(16) << loss.value;

		layer2.backward(norm.output, loss.diff);
		norm.backward(activation.output, layer2.diff);
		activation.backward(layer1.output, norm.diff);
		layer1.backward(in, activation.diff);

		

		layer1.forward(in);
		activation.forward(layer1.output);
		norm.forward(activation.output, false);
		layer2.forward(norm.output);
		loss.calculate(layer2.output, out);
		cout << setw(16) << loss.value << endl;


		weights1.step(layer1.weights, layer1.weightsDiff);
		biases1.step(layer1.biases, layer1.biasesDiff);
		weights2.step(layer2.weights, layer2.weightsDiff);
		biases2.step(layer2.biases, layer2.biasesDiff);
		gamma.step(norm.gamma, norm.gammaDiff);
		beta.step(norm.beta, norm.betaDiff);
		layer1.zeroGradients();
		layer2.zeroGradients();
		norm.zeroGradients();

		//cout << setw(10) << norm.u1[0] << setw(10) << norm.u2[0] << setw(10) << norm.count << "\n\n";

		/*layer1.forward(in);
		activation.forward(layer1.output);
		layer2.forward(activation.output);
		loss.calculate(layer2.output, out);

		cout << setw(16) << loss.value;

		layer2.backward(activation.output, loss.diff);
		activation.backward(layer1.output, layer2.diff);
		layer1.backward(in, activation.diff);

		weights1.step(layer1.weights, layer1.weightsDiff);
		biases1.step(layer1.biases, layer1.biasesDiff);
		weights2.step(layer2.weights, layer2.weightsDiff);
		biases2.step(layer2.biases, layer2.biasesDiff);
		layer1.zeroGradients();
		layer2.zeroGradients();

		layer1.forward(in);
		activation.forward(layer1.output);
		layer2.forward(activation.output);
		loss.calculate(layer2.output, out);
		cout << setw(16) << loss.value << endl;*/
	}


	return 0;
}