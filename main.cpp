#include <iostream>
#include <string>
#include <iomanip>
#include <filesystem>
#include "CNN.h"

using namespace std;



typedef vector<double> v;
typedef vector<vector<double>> v2;
typedef vector<vector<vector<double>>> v3;
typedef vector<vector<vector<vector<double>>>> v4;
typedef vector<vector<vector<vector<vector<double>>>>> v5;

int main() {
	srand(1);

	cout.precision(8);

	string trainFolderName = "train/";
	string testFolderName = "test/";
	string saveFolderName = "models/";

	int nOfTrainSamples = 60000;
	int nOfTestSamples = 10000;

	int nOfClasses = 10;

	vector<int> imageSize = { 1, 28, 28 };

	int batchSize = 100;

	vector<int> batchesOrder(nOfTrainSamples);

	int epochs = 10;

	CNN cnn;

	v4 input(batchSize, v3(imageSize[0], v2(imageSize[1], v(imageSize[2]))));
	vector<int> targets(batchSize);

	for (int e = 1; e <= epochs; e++) {

		for (int i = 0; i < nOfTrainSamples; i++) {
			batchesOrder[i] = i;
		}
		for (int i = 0; i < nOfTrainSamples; i++) {
			int k = rand() % (nOfTrainSamples - i);
			swap(batchesOrder[i], batchesOrder[i + k]);
		}

		int count = 0;

		for (int leftIndex = 0; leftIndex < nOfTrainSamples; leftIndex += batchSize) {
			int rightIndex = min(leftIndex + batchSize, nOfTrainSamples);
			int size = rightIndex - leftIndex;

			for (int sampleIndex = leftIndex; sampleIndex < rightIndex; sampleIndex++) {
				string fileName = to_string(batchesOrder[sampleIndex]) + ".txt";

				ifstream inputFile(trainFolderName + fileName);

				inputFile >> targets[sampleIndex - leftIndex];
				for (int i = 0; i < imageSize[0]; i++) {
					for (int j = 0; j < imageSize[1]; j++) {
						for (int k = 0; k < imageSize[2]; k++) {
							inputFile >> input[sampleIndex - leftIndex][i][j][k];
						}
					}
				}
				inputFile.close();
			}

			cnn.forward(input, size, true);
			cout << "Epoch " << e << ", batch [" << leftIndex << "; " << rightIndex << "), " <<
				"loss = " << cnn.getLoss();
			vector<vector<double>> output = cnn.getOutputs();
			for (int i = 0; i < size; i++) {
				if (output[i][targets[i]] > 0.5) {
					count++;
				}
			}
			cout << ", accuracy = " << (double)count / (double)rightIndex << endl;

			cnn.backward(input, targets, size);
			cnn.updateParameters();

			if (leftIndex % 1000 == 0) {
				string folderName = "model" + to_string(e) + "-" + to_string(leftIndex);
				filesystem::create_directory(saveFolderName + folderName);
				cnn.save(saveFolderName + folderName + "/");
			}
 		}

	}

	return 0;
}