#include <iostream>
#include <time.h>
#include "layers.h"

int main() {
	long long start, end;
	int b = 1000;
	int d = 800;
	LinearLayer layer(d, d, b, 8);
	vector<vector<double>> in;
	in.resize(b);
	for (int i = 0; i < b; i++) {
		in[i].resize(d);
	}
	start = clock();
	layer.forward(in);
	end = clock();
	double t = (double)(end - start) / (double)(CLOCKS_PER_SEC);
	cout << "Time: " << t << "s\n";
	return 0;
}