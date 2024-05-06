#pragma once

#include <iostream>
#include <vector>

using namespace std;

class MSELoss {
public:
	int nOfOutputs = 0, maxBatchSize = 0;
	double value = 0;
	vector<vector<double>> diff;

public:
	MSELoss(int outputs, int batchSize);
	void calculate(vector<vector<double>>& output,
		vector<vector<double>>& target);
};


class CategoricalCrossentropyLoss {
public:
	int nOfClasses = 0, maxBatchSize = 0;
	double value = 0;
	vector<vector<double>> diff;
public:
	CategoricalCrossentropyLoss(int n, int batchSize);
	void calculate(vector<vector<double>>& output, vector<int>& target);
};