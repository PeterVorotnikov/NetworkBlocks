#include <iostream>
#include "layers.h"
#include "lossFunctions.h"
#include "optimizers.h"

int main() {
	ConvolutionalLayer layer(10, 10, 2, 3, 1);
	vector<vector<vector<vector<double>>>> in(1, vector<vector<vector<double>>>(2,
		vector < vector < double>>(10, vector<double>(10, 1))));
	layer.forward(in);

	return 0;
}