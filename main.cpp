#include <iostream>
#include <iomanip>
#include "CNN.h"



typedef vector<double> v;
typedef vector<vector<double>> v2;
typedef vector<vector<vector<double>>> v3;
typedef vector<vector<vector<vector<double>>>> v4;
typedef vector<vector<vector<vector<vector<double>>>>> v5;

int main() {
	cout.precision(5);

	CNN cnn;

	int batchSize = 2;

	v4 images(batchSize, v3(1, v2(28, v(28, 0.5))));
	for (int b = 0; b < batchSize; b++) {
		for (int r = 0; r < 28; r++) {
			for (int c = 0; c < 28; c++) {
				images[b][0][r][c] = (double)rand() / (double)RAND_MAX;
			}
		}
	}
	vector<int> targets(batchSize);
	for (int i = 0; i < batchSize; i++) {
		targets[i] = rand() % 10;
	}

	

	for (int e = 0; e < 1000; e++) {
		cnn.forward(images, true);
		cnn.backward(images, targets);
		double loss = cnn.getLoss();
		cnn.updateParameters();
		cout << "e = " << e + 1 << endl;
		cout << "loss = " << loss << endl;
		cout << "target = " << targets[1] << endl;
		cnn.forward(images, false);
		vector<vector<double>> outputs = cnn.getOutputs();
		for (int i = 0; i < 10; i++) {
			cout << outputs[1][i] << " ";
		}
		cout << "\n\n";
	}

	return 0;
}