#include <iostream>
#include <iomanip>
#include <thread>
#include "layers.h"
#include "lossFunctions.h"
#include "optimizers.h"

void print(vector<double>& v) {
	for (int i = 0; i < v.size(); i++) {
		cout << setw(6) << v[i];
	}
	cout << endl;
}

typedef vector<double> v;
typedef vector<vector<double>> v2;
typedef vector<vector<vector<double>>> v3;
typedef vector<vector<vector<vector<double>>>> v4;
typedef vector<vector<vector<vector<vector<double>>>>> v5;

void f(vector<int>* v) {
	for (int i = 0; i < v->size(); i++) {
		(*v)[i] *= 2;
	}
}

int main() {
	int batches = 100, sizeIn = 200, sizeOut = 10;
	vector<vector<double>> in(batches, vector<double>(sizeIn));
	vector<vector<double>> out(batches, vector<double>(sizeIn));
	for (int b = 0; b < batches; b++) {
		for (int i = 0; i < sizeIn; i++) {
			in[b][i] = (double)rand() / (double)RAND_MAX;
		}
	}

	for (int b = 0; b < batches; b++) {
		for (int i = 0; i < sizeOut; i++) {
			out[b][i] = (double)rand() / (double)RAND_MAX;
		}
	}

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


	layer1.load("model/layer1.txt");
	norm.load("model/norm.txt");
	layer2.load("model/layer2.txt");
	weights1.load("model/weights1.txt");
	weights2.load("model/weights2.txt");
	biases1.load("model/biases1.txt");
	biases2.load("model/biases2.txt");
	gamma.load("model/gamma.txt");
	beta.load("model/beta.txt");

	for (int e = 0; e < 100; e++) {
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

		
	}

	layer1.save("model/layer1.txt");
	norm.save("model/norm.txt");
	layer2.save("model/layer2.txt");
	weights1.save("model/weights1.txt");
	weights2.save("model/weights2.txt");
	biases1.save("model/biases1.txt");
	biases2.save("model/biases2.txt");
	gamma.save("model/gamma.txt");
	beta.save("model/beta.txt");


	return 0;
}