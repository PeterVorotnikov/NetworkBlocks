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

class ReLU1d {
public:
	int d = 0, l = 0, maxBatchSize = 0;
	vector<vector<double>> output;
	vector<vector<double>> diff;

public:
	ReLU1d(int d, int batchSize, int l = 0);
	void forward(vector<vector<double>>& input);
	void backward(vector<vector<double>>& input, vector<vector<double>>& nextDiff);
};

class Sigmoid1d {
public:
	int d = 0, maxBatchSize = 0;
	vector<vector<double>> output;
	vector<vector<double>> diff;
	double f(double x) {
		return 1.0 / (1.0 + exp(-x));
	}

public:
	Sigmoid1d(int d, int batchSize);
	void forward(vector<vector<double>>& input);
	void backward(vector<vector<double>>& input, vector<vector<double>>& nextDiff);
};


class ReLU3d {
public:
	int d1 = 0, d2 = 0, d3 = 0, l = 0, maxBatchSize = 0;
	vector<vector<vector<vector<double>>>> output;
	vector<vector<vector<vector<double>>>> diff;

public:
	ReLU3d(int d1, int d2, int d3, int batchSize, int l = 0);
	void forward(vector<vector<vector<vector<double>>>>& input);
	void backward(vector<vector<vector<vector<double>>>>& input, 
		vector<vector<vector<vector<double>>>>& nextDiff);
};