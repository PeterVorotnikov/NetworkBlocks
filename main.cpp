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

typedef vector<double> v;
typedef vector<vector<double>> v2;
typedef vector<vector<vector<double>>> v3;
typedef vector<vector<vector<vector<double>>>> v4;
typedef vector<vector<vector<vector<vector<double>>>>> v5;

int main() {
	cout.precision(5);
	
	int batchSize = 10, d1 = 3, d2 = 10, d3 = 10, dOut = 10;
	int nOfBatches = 5;
	v5 in(nOfBatches, v4(batchSize, v3(d1, v2(d2, v(d3)))));
	v3 out(nOfBatches, v2(batchSize, v(dOut)));
	for (int k = 0; k < nOfBatches; k++) {
		for (int b = 0; b < batchSize; b++) {
			for (int i1 = 0; i1 < d1; i1++) {
				for (int i2 = 0; i2 < d2; i2++) {
					for (int i3 = 0; i3 < d3; i3++) {
						in[k][b][i1][i2][i3] = (double)rand() / (double)RAND_MAX;
					}
				}
			}
			for (int i = 0; i < dOut; i++) {
				out[k][b][i] = (double)rand() / (double)RAND_MAX;
			}
		}
	}

	ConvolutionalLayer conv(d2, d3, d1, d1, 100, 5);
	ReLU3d a1(d1, d2, d3, 100);
	BatchNormalization3d norm1(d1, d2, d3, 100);
	Flatten31 flatten(d1, d2, d3, 100);
	LinearLayer linear(flatten.outputSize, dOut, 100);
	Sigmoid1d a2(dOut, 100);

	MSELoss loss(dOut, 100);

	Adam4d convW(d1, d1, 5, 5);
	Adam1d convB(d1);
	Adam3d norm1G(d1, d2, d3);
	Adam3d norm1B(d1, d2, d3);
	Adam2d linearW(flatten.outputSize, dOut);
	Adam1d linearB(dOut);

	for (int e = 0; e < 1000; e++) {

		cout << "Epoch " << e + 1 << "\n\n";

		for (int k = 0; k < nOfBatches; k++) {

			/*conv.forward(in[k]);
			a1.forward(conv.output);
			flatten.forward(a1.output);
			linear.forward(flatten.output);
			a2.forward(linear.output);
			loss.calculate(a2.output, out[k]);

			cout << "s = " << k + 1 << ", loss = " << loss.value << endl;

			a2.backward(linear.output, loss.diff);
			linear.backward(flatten.output, a2.diff);
			flatten.backward(a1.output, linear.diff);
			a1.backward(conv.output, flatten.diff);
			conv.backward(in[k], a1.diff);

			convW.step(conv.weights, conv.weightsDiff);
			convB.step(conv.biases, conv.biasesDiff);
			linearW.step(linear.weights, linear.weightsDiff);
			linearB.step(linear.biases, linear.biasesDiff);

			linear.zeroGradients();
			conv.zeroGradients();*/



			conv.forward(in[k]);
			a1.forward(conv.output);
			norm1.forward(a1.output);
			flatten.forward(norm1.output);
			linear.forward(flatten.output);
			a2.forward(linear.output);
			loss.calculate(a2.output, out[k]);

			cout << "s = " << k + 1 << ", loss = " << loss.value << "   ";

			a2.backward(linear.output, loss.diff);
			linear.backward(flatten.output, a2.diff);
			flatten.backward(norm1.output, linear.diff);
			norm1.backward(a1.output, flatten.diff);
			a1.backward(conv.output, norm1.diff);
			conv.backward(in[k], a1.diff);

			convW.step(conv.weights, conv.weightsDiff);
			convB.step(conv.biases, conv.biasesDiff);
			linearW.step(linear.weights, linear.weightsDiff);
			linearB.step(linear.biases, linear.biasesDiff);
			norm1B.step(norm1.beta, norm1.betaDiff);
			norm1G.step(norm1.gamma, norm1.gammaDiff);

			linear.zeroGradients();
			conv.zeroGradients();
			norm1.zeroGradients();

			conv.forward(in[k]);
			a1.forward(conv.output);
			norm1.forward(a1.output, false);
			flatten.forward(norm1.output);
			linear.forward(flatten.output);
			a2.forward(linear.output);
			loss.calculate(a2.output, out[k]);

			cout << "test loss = " << loss.value << endl;

		}

		cout << "\n\n";

	}
	

	return 0;
}