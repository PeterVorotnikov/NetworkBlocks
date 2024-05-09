#pragma once

#include <fstream>
#include <vector>

using namespace std;

class Adam1d {
public:
	int d = 0;
	double alpha = 0, beta1 = 0, beta2 = 0, l2 = 0;
	double epsilon = pow(10, -8);
	vector<double> v;
	vector<double> g;

public:
	Adam1d(int d, double alpha = 0.0003, double beta1 = 0.9, double beta2 = 0.99, double l2 = 0) {
		this->d = d;
		this->alpha = alpha;
		this->beta1 = beta1;
		this->beta2 = beta2;
		this->l2 = l2;
		v.resize(d);
		g.resize(d);
	}
	void step(vector<double>& parameters, vector<double>& diff) {
		for (int i = 0; i < d; i++) {
			v[i] = beta1 * v[i] + (1 - beta1) * (diff[i] + 2 * l2 * parameters[i]);
			g[i] = beta2 * g[i] + (1 - beta2) * pow(diff[i], 2);
			parameters[i] -= alpha / sqrt(g[i] + epsilon) * v[i];
		}
	}
	void save(string fileName) {
		ofstream file(fileName);
		for (int i = 0; i < d; i++) {
			file << v[i] << " " << g[i] << " ";
		}
		file.close();
	}
	void load(string fileName) {
		ifstream file(fileName);
		for (int i = 0; i < d; i++) {
			file >> v[i] >> g[i];
		}
		file.close();
	}
};


class Adam2d {
public:
	int d1 = 0, d2 = 0;
	double alpha = 0, beta1 = 0, beta2 = 0, l2 = 0;
	double epsilon = pow(10, -8);
	vector<vector<double>> v;
	vector<vector<double>> g;

public:
	Adam2d(int d1, int d2, double alpha = 0.0003, double beta1 = 0.9, double beta2 = 0.99, 
		double l2 = 0) {
		this->d1 = d1;
		this->d2 = d2;
		this->alpha = alpha;
		this->beta1 = beta1;
		this->beta2 = beta2;
		this->l2 = l2;
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
				v[i1][i2] = beta1 * v[i1][i2] + (1 - beta1) * (diff[i1][i2] + 
					2 * l2 * parameters[i1][i2]);
				g[i1][i2] = beta2 * g[i1][i2] + (1 - beta2) * pow(diff[i1][i2], 2);
				parameters[i1][i2] -= alpha / sqrt(g[i1][i2] + epsilon) * v[i1][i2];
			}
		}
	}
	void save(string fileName) {
		ofstream file(fileName);
		for (int i1 = 0; i1 < d1; i1++) {
			for (int i2 = 0; i2 < d2; i2++) {
				file << v[i1][i2] << " " << g[i1][i2] << " ";
			}
		}
		file.close();
	}
	void load(string fileName) {
		ifstream file(fileName);
		for (int i1 = 0; i1 < d1; i1++) {
			for (int i2 = 0; i2 < d2; i2++) {
				file >> v[i1][i2] >> g[i1][i2];
			}
		}
		file.close();
	}
};



class Adam3d {
public:
	int d1 = 0, d2 = 0, d3 = 0;
	double alpha = 0, beta1 = 0, beta2 = 0, l2 = 0;
	double epsilon = pow(10, -8);
	vector<vector<vector<double>>> v;
	vector<vector<vector<double>>> g;

public:
	Adam3d(int d1, int d2, int d3, double alpha = 0.0003, double beta1 = 0.9,
		double beta2 = 0.99, double l2 = 0) {
		this->d1 = d1;
		this->d2 = d2;
		this->d3 = d3;
		this->alpha = alpha;
		this->beta1 = beta1;
		this->beta2 = beta2;
		this->l2 = l2;

		v.resize(d1);
		g.resize(d1);
		for (int i = 0; i < d1; i++) {
			v[i].resize(d2);
			g[i].resize(d2);
			for (int j = 0; j < d2; j++) {
				v[i][j].resize(d3);
				g[i][j].resize(d3);
			}
		}
	}
	void step(vector<vector<vector<double>>>& parameters,
		vector<vector<vector<double>>>& diff) {
		for (int i1 = 0; i1 < d1; i1++) {
			for (int i2 = 0; i2 < d2; i2++) {
				for (int i3 = 0; i3 < d3; i3++) {
					v[i1][i2][i3] = beta1 * v[i1][i2][i3] + (1 - beta1) *
						(diff[i1][i2][i3] + 2 * l2 * parameters[i1][i2][i3]);
					g[i1][i2][i3] = beta2 * g[i1][i2][i3] + (1 - beta2) *
						pow(diff[i1][i2][i3], 2);
					parameters[i1][i2][i3] -= alpha / sqrt(g[i1][i2][i3] + epsilon) *
						v[i1][i2][i3];
				}
			}
		}
	}
	void save(string fileName) {
		ofstream file(fileName);
		for (int i1 = 0; i1 < d1; i1++) {
			for (int i2 = 0; i2 < d2; i2++) {
				for (int i3 = 0; i3 < d3; i3++) {
					file << v[i1][i2][i3] << " " << g[i1][i2][i3] << " ";
				}
			}
		}
		file.close();
	}
	void load(string fileName) {
		ifstream file(fileName);
		for (int i1 = 0; i1 < d1; i1++) {
			for (int i2 = 0; i2 < d2; i2++) {
				for (int i3 = 0; i3 < d3; i3++) {
					file >> v[i1][i2][i3] >> g[i1][i2][i3];
				}
			}
		}
		file.close();
	}
};



class Adam4d {
public:
	int d1 = 0, d2 = 0, d3 = 0, d4 = 0;
	double alpha = 0, beta1 = 0, beta2 = 0, l2 = 0;
	double epsilon = pow(10, -8);
	vector<vector<vector<vector<double>>>> v;
	vector<vector<vector<vector<double>>>> g;

public:
	Adam4d(int d1, int d2, int d3, int d4, double alpha = 0.0003, double beta1 = 0.9, 
		double beta2 = 0.99, double l2 = 0) {
		this->d1 = d1;
		this->d2 = d2;
		this->d3 = d3;
		this->d4 = d4;
		this->alpha = alpha;
		this->beta1 = beta1;
		this->beta2 = beta2;
		this->l2 = l2;

		v.resize(d1);
		g.resize(d1);
		for (int i = 0; i < d1; i++) {
			v[i].resize(d2);
			g[i].resize(d2);
			for (int j = 0; j < d2; j++) {
				v[i][j].resize(d3);
				g[i][j].resize(d3);
				for (int k = 0; k < d3; k++) {
					v[i][j][k].resize(d4);
					g[i][j][k].resize(d4);
				}
			}
		}
	}
	void step(vector<vector<vector<vector<double>>>>& parameters, 
		vector<vector<vector<vector<double>>>>& diff) {
		for (int i1 = 0; i1 < d1; i1++) {
			for (int i2 = 0; i2 < d2; i2++) {
				for (int i3 = 0; i3 < d3; i3++) {
					for (int i4 = 0; i4 < d4; i4++) {
						v[i1][i2][i3][i4] = beta1 * v[i1][i2][i3][i4] + (1 - beta1) *
							(diff[i1][i2][i3][i4] + 2 * l2 * parameters[i1][i2][i3][i4]);
						g[i1][i2][i3][i4] = beta2 * g[i1][i2][i3][i4] + (1 - beta2) *
							pow(diff[i1][i2][i3][i4], 2);
						parameters[i1][i2][i3][i4] -= alpha / sqrt(g[i1][i2][i3][i4] + epsilon) *
							v[i1][i2][i3][i4];
					}
				}
			}
		}
	}
	void save(string fileName) {
		ofstream file(fileName);
		for (int i1 = 0; i1 < d1; i1++) {
			for (int i2 = 0; i2 < d2; i2++) {
				for (int i3 = 0; i3 < d3; i3++) {
					for (int i4 = 0; i4 < d4; i4++) {
						file << v[i1][i2][i3][i4] << " " << g[i1][i2][i3][i4] << " ";
					}
				}
			}
		}
		file.close();
	}
	void load(string fileName) {
		ifstream file(fileName);
		for (int i1 = 0; i1 < d1; i1++) {
			for (int i2 = 0; i2 < d2; i2++) {
				for (int i3 = 0; i3 < d3; i3++) {
					for (int i4 = 0; i4 < d4; i4++) {
						file >> v[i1][i2][i3][i4] >> g[i1][i2][i3][i4];
					}
				}
			}
		}
		file.close();
	}
};