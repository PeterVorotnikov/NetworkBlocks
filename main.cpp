#include <iostream>
#include <time.h>
#include <mutex>
#include "layers.h"

mutex mtx;

void f(vector<int>& v, int index) {
	int val = v[index];
	for (int i = 0; i < 200000000; i++) {
		val++;
	}
	v[index] = val;
}


int main() {
	vector<int> v = { 0, 0, 0, 0 };
	vector<thread> threads;
	for (int i = 0; i < 4; i++) {
		thread t(f, ref(v), i);
		threads.push_back(move(t));
	}
	for (int i = 0; i < 4; i++) {
		threads[i].join();
	}
	for (int i = 0; i < v.size(); i++) {
		cout << v[i] << " ";
	}
	cout << endl;
}