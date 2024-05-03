#include <iostream>
#include "layers.h"

int main() {
	LinearLayer layer(2, 2, 2);
	vector<vector<double>> in = { {-1, 0.5}, {0, 0.1} };
	vector<vector<double>> diff = { {1, 1}, {1, 1} };
	layer.forward(in);
	layer.backward(in, diff);
	return 0;
}