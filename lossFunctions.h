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


class CathegoricalCrossentropyLoss {
public:
	int nOfClasses = 0, maxBatchSize = 0;
	double value = 0;
	vector<vector<double>> diff;
public:
	CathegoricalCrossentropyLoss(int n, int batchSize);
	void calculate(vector<vector<double>>& output, vector<int>& target);
};