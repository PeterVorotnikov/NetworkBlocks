#pragma once

#include <vector>

using namespace std;

class Adam1d {
public:
	int d = 0;
	double alpha = 0, beta1 = 0, beta2 = 0;
	double epsilon = pow(10, -8);
	vector<double> v;
	vector<double> g;

public:
	Adam1d(int d, double alpha = 0.0003, double beta1 = 0.9, double beta2 = 0.99) {
		this->d = d;
		this->alpha = alpha;
		this->beta1 = beta1;
		this->beta2 = beta2;
		v.resize(d);
		g.resize(d);
	}
	void step(vector<double>& parameters, vector<double>& diff) {
		for (int i = 0; i < d; i++) {
			v[i] = beta1 * v[i] + (1 - beta1) * diff[i];
			g[i] = beta2 * g[i] + (1 - beta2) * pow(diff[i], 2);
			parameters[i] -= alpha / sqrt(g[i] + epsilon) * v[i];
		}
	}
};


class Adam2d {
public:
	int d1 = 0, d2 = 0;
	double alpha = 0, beta1 = 0, beta2 = 0;
	double epsilon = pow(10, -8);
	vector<vector<double>> v;
	vector<vector<double>> g;

public:
	Adam2d(int d1, int d2, double alpha = 0.0003, double beta1 = 0.9, double beta2 = 0.99) {
		this->d1 = d1;
		this->d2 = d2;
		this->alpha = alpha;
		this->beta1 = beta1;
		this->beta2 = beta2;
		v.resize(d1);
		g.resize(d1);
		for (int i = 0; i < d1; i++) {
			v[i].resize(d2);
			g[i].resize(d2);
		}
	}
	void step(vector<vector<double>>& parameters, vector<vector<double>>& diff) {
		for (int i1 = 0; i1 < d1; i1++) {
			for (int i2 = 0; i2 < d2; i2++) {
				v[i1][i2] = beta1 * v[i1][i2] + (1 - beta1) * diff[i1][i2];
				g[i1][i2] = beta2 * g[i1][i2] + (1 - beta2) * pow(diff[i1][i2], 2);
				parameters[i1][i2] -= alpha / sqrt(g[i1][i2] + epsilon) * v[i1][i2];
			}
		}
	}
};