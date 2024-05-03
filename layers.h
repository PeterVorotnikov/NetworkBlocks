#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <cmath>

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

class ReLU2d {
public:
	int d = 0, l = 0, maxBatchSize = 0;
	vector<vector<double>> output;
	vector<vector<double>> diff;

public:
	ReLU2d(int d, int batchSize, int l = 0);
	void forward(vector<vector<double>>& input);
	void backward(vector<vector<double>>& input, vector<vector<double>>& nextDiff);
};

class Sigmoid {
public:
	int d = 0, maxBatchSize = 0;
	vector<vector<double>> output;
	vector<vector<double>> diff;
	double f(double x) {
		return 1.0 / (1.0 + exp(-x));
	}

public:
	Sigmoid(int d, int batchSize);
	void forward(vector<vector<double>>& input);
	void backward(vector<vector<double>>& input, vector<vector<double>>& nextDiff);
};