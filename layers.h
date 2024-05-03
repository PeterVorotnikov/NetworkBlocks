#pragma once

#include <iostream>
#include <vector>
#include <random>

using namespace std;


class LinearLayer {
public:
	int nOfInputs = 0, nOfOutputs = 0, maxBatchSize = 0;

	vector<vector<double>> output;
	vector<vector<double>> diff;

	vector<vector<double>> weights;
	vector<vector<double>> weightsDiff;
	vector<double> biases;
	vector<double> biasesDiff;

private:
	void init();
	void initStates();
	void initWeights();

public:
	LinearLayer(int inputs, int outputs, int batchSize);
	void forward(vector<vector<double>>& input);
	void backward(vector<vector<double>>& input, vector<vector<double>>& nextDiff);
	void zeroGradients();
};