#include <iostream>
#include "layers.h"

int main() {
	LinearLayer layer(2, 2, 2, 4);
	vector<vector<double>> in = { {-1, 1}, {1, -1} };
	layer.forward(in);
	return 0;
}