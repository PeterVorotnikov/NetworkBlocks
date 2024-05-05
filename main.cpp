#include <iostream>
#include <iomanip>
#include "layers.h"
#include "lossFunctions.h"
#include "optimizers.h"

int main() {
	srand(1);


	int batchSize = 1, channels = 1, rows = 6, cols = 6;
	MaxPooling pool(channels, rows, cols, batchSize);
	vector<vector<vector<vector<double>>>> in(batchSize,
		vector<vector<vector<double>>>(channels,
			vector<vector<double>>(rows, vector<double>(cols))));
	vector<vector<vector<vector<double>>>> nextDiff(batchSize,
		vector<vector<vector<double>>>(channels,
			vector<vector<double>>(rows / 2, vector<double>(cols / 2, 0.5))));
	
	for (int test = 0; test < 10; test++) {


		

		for (int i1 = 0; i1 < in.size(); i1++) {
			for (int i2 = 0; i2 < in[i1].size(); i2++) {
				for (int i3 = 0; i3 < in[i1][i2].size(); i3++) {
					for (int i4 = 0; i4 < in[i1][i2][i3].size(); i4++) {
						double r = (double)rand() / (double)RAND_MAX;
						in[i1][i2][i3][i4] = r;
					}
				}
			}
		}
		pool.forward(in);
		pool.backward(in, nextDiff);
		for (int i1 = 0; i1 < batchSize; i1++) {

			cout << "Input\n";
			for (int i2 = 0; i2 < channels; i2++) {
				for (int i3 = 0; i3 < rows; i3++) {
					for (int i4 = 0; i4 < cols; i4++) {
						cout << setw(10) << in[i1][i2][i3][i4];
					}
					cout << endl;
				}
			}
			cout << "\nOutput\n";
			for (int i2 = 0; i2 < channels; i2++) {
				for (int i3 = 0; i3 < rows / 2; i3++) {
					for (int i4 = 0; i4 < cols / 2; i4++) {
						cout << setw(10) << pool.output[i1][i2][i3][i4];
					}
					cout << endl;
				}
			}
			cout << "\Diff\n";
			for (int i2 = 0; i2 < channels; i2++) {
				for (int i3 = 0; i3 < rows; i3++) {
					for (int i4 = 0; i4 < cols; i4++) {
						cout << setw(10) << pool.diff[i1][i2][i3][i4];
					}
					cout << endl;
				}
			}
			cout << "\Memory\n";
			for (int i2 = 0; i2 < channels; i2++) {
				for (int i3 = 0; i3 < rows / 2; i3++) {
					for (int i4 = 0; i4 < cols / 2; i4++) {
						cout << setw(10) << pool.memory[i1][i2][i3][i4];
					}
					cout << endl;
				}
			}
		}

	}

	return 0;
}