#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/saturate.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <vector>
#include "Feature_Map.h"
#include <sstream>
#include <omp.h>
#include <chrono>
#include <ctime>    

#define detectionRateTarget 0.995
#define falsePositiveTarget 0.5
#define overallTargetFalsePositiveRate 1e-6

using namespace cv;
using namespace std;
using namespace Eigen;

typedef Matrix <long double, Dynamic, 1> VectorXld;


int a[25][25];//defalt frame size r = 480, c = 640 , mul factor = 0.9

int tem = -1, w, h, nRows = 24, nCols = 24;
float m;
string path, first, second, third;

vector <pair <int, int>> Feature_vector;
struct StumpRule
{
	int featureIndex;
	double threshold;
	double margin;
	double weightedError;
	int toggle;
};

int sumint(int z_1, int z_2, int z_3, int z_4)
{
	return (a[z_2 + z_4 - 1][z_1 + z_3 - 1] + a[z_2 - 1][z_1 - 1] - a[z_2 + z_4 - 1][z_1 - 1] - a[z_2 - 1][z_1 + z_3 - 1]);
}
bool detect(Mat& img, vector<StumpRule>*& cascade, VectorXf& shifts, int round);
int Compute_Feature(int z);



class TrainDataSet
{
	int nPositives;
	int nNegatives;
	int sampleCount;
	int round;
	//int exampleIndex;
	int posAverageWeight;
	int negAverageWeight;
	double initialPositiveWeight;
	double positiveTotalWeight;
	double negativeTotalWeight;
	VectorXi negatives;
	VectorXld weights;
	VectorXi labels;
	int featureCount;
	bool TrainMode;


public:
	vector <StumpRule> committee;
	TrainDataSet(vector<StumpRule>*& cascade, VectorXf& shifts, int r, bool Train)
	{

		nPositives = 1000;
		nNegatives = 1000;
		sampleCount = nPositives + nNegatives;
		if (round == 0)
			negatives.setZero(1000);
		//weights.setZero(sampleCount);
		labels.setZero(sampleCount);
		round = r;
		TrainMode = Train;
		if (TrainMode) {
			positiveTotalWeight = 0.5;
			negativeTotalWeight = 1.0 - positiveTotalWeight;
			weights.setZero(sampleCount);
			double posAverageWeight = positiveTotalWeight / nPositives;
			double negAverageWeight = negativeTotalWeight / nNegatives;
			featureCount = 162336;
			for (int i = 0; i < sampleCount; i++)
				weights(i) = i < nPositives ? posAverageWeight : negAverageWeight;
		}

		for (int i = 0; i < sampleCount; i++)
			labels(i) = i < nPositives ? 1 : -1;

		int i = 0;
		//#pragma omp parallel for schedule(static)
		for (int n = 0; i < 1000; n++) {
			if (TrainMode)
				first = "neg/neg (";
			else
				first = "vNeg/vNeg (";
			if (round == 0)
				second = to_string(n);
			else
				second = n < 1000 ? to_string(negatives(n)) : to_string(n + negatives(999));//---------------
			third = ").pgm";
			path = first + second + third;
			Mat img = imread(path, 0), mean, stddev;
			meanStdDev(
				img,
				mean,
				stddev
			);

			if (round == 0 || n < negatives(999) || stddev.at<double>(0, 0) > 1) {

				if (detect(img, cascade, shifts, round)) {
					negatives(i) = n;
					i++;
				}
			}

		}


	}

	pair <int, int> getFeature_Index(int featureIndex, int iterator)
	{
		if (featureIndex == tem)
			return Feature_vector[iterator];
		else {
			tem = featureIndex;
			Feature_vector.clear();

			#pragma omp parallel for schedule(static)
			for (int p_c = 0; p_c < 2000; p_c++) {
				string tfeature;
				int feature;
				if (p_c < 1000)
					first = "Features/" + to_string(p_c) + ".txt";
				else {
					int dum = p_c - 1000;
					first = "Features/" + to_string(negatives(dum) + 1000) + ".txt";
				}
				//cout << first<<endl;
				ifstream File_1(first);

				File_1.seekg(featureIndex * 7);
				getline(File_1, tfeature);
				stringstream geek(tfeature);
				geek >> feature;
				#pragma omp critical
				Feature_vector.push_back(make_pair(feature, p_c));

				File_1.close();
			}
			sort(Feature_vector.begin(), Feature_vector.end());
			return Feature_vector[iterator];
		}
	}



	void descisionStump(int featureIndex, StumpRule& best)
	{
		best.featureIndex = featureIndex;
		best.margin = 0;
		best.threshold = getFeature_Index(featureIndex, 0).first - 1.0;
		best.weightedError = 2;
		best.toggle = 0;

		StumpRule temp = best;
		double pErr;
		double nErr;
		double bpositiveWeight = positiveTotalWeight;
		double bnegativeWeight = 1 - positiveTotalWeight;
		double spositiveWeight = 0;
		double snegativeWeight = 0;

		int iterator = 0;

		while (true)
		{
			pErr = spositiveWeight + bnegativeWeight;
			nErr = bpositiveWeight + snegativeWeight;

			if (pErr < nErr)
			{
				temp.weightedError = pErr;
				temp.toggle = 1;
			}
			else
			{
				temp.weightedError = nErr;
				temp.toggle = -1;
			}

			if (stumpOrder(temp, best))
			{
				best = temp;
				//cout << "bh " << best.threshold << endl;
			}

			if (iterator == sampleCount - 1)
				break;

			iterator++;

			while (true)
			{
				int sampleIndex = getFeature_Index(featureIndex, iterator).second;

				//cout << endl << "sampleIndex = " << sampleIndex<<endl;

				if (labels[sampleIndex] < 0)
				{
					snegativeWeight = snegativeWeight + weights[sampleIndex];
					bnegativeWeight = bnegativeWeight - weights[sampleIndex];
				}
				else
				{
					spositiveWeight = spositiveWeight + weights[sampleIndex];
					bpositiveWeight = bpositiveWeight - weights[sampleIndex];
				}

				if (iterator == sampleCount - 1 || getFeature_Index(featureIndex, iterator).first != getFeature_Index(featureIndex, iterator + 1).first)
					break;
				else
					iterator++;
			}

			if (iterator == sampleCount - 1)
			{
				temp.threshold = getFeature_Index(featureIndex, iterator).first + 1.0;
				temp.margin = 0;
			}
			else
			{
				temp.threshold = (getFeature_Index(featureIndex, iterator).first + getFeature_Index(featureIndex, iterator + 1).first) / 2;
				temp.margin = getFeature_Index(featureIndex, iterator + 1).first - getFeature_Index(featureIndex, iterator).first;
			}

			//cout << "TH " << temp.threshold << " , it " << iterator << endl;

		}

	}

	bool stumpOrder(StumpRule& s1, StumpRule& s2)
	{

		if ((s1.weightedError < s2.weightedError) || ((s1.weightedError == s2.weightedError) && s1.margin > s2.margin))
			return true;
		else
			return false;
	}

	StumpRule bestStump()
	{

		StumpRule temp, best;
		descisionStump(0, best);
		/*
		for (int i = 0;i < 2000;i++)
			cout << Feature_vector[i].first << " " << Feature_vector[i].second << endl;
		*/
#pragma omp parallel for schedule(static)
		for (int f = 1; f < featureCount; f++)
		{
			descisionStump(f, temp);
			if (stumpOrder(temp, best))
				best = temp;
		}

		return best;
	}

	void adaboost()
	{

		StumpRule rule = bestStump();

		committee.push_back(rule);

		VectorXi prediction(sampleCount);
		//#pragma omp parallel for schedule(static)
		for (int i = 0; i < sampleCount; i++)
		{
			int sampleIndex = getFeature_Index(rule.featureIndex, i).second;
			int temp = (getFeature_Index(rule.featureIndex, i).first > rule.threshold ? 1 : -1) * rule.toggle;
			prediction[sampleIndex] = temp > 0 ? 1 : -1;
		}

		VectorXi agree = labels.cwiseProduct(prediction);

		//update weights
		VectorXld weightUpdate;
		weightUpdate.setOnes(sampleCount);
		bool errorFlag = false;

		for (int exampleIndex = 0; exampleIndex < sampleCount; exampleIndex++)
		{
			//more weight for a difficult example
			if (agree[exampleIndex] < 0)
			{
				weightUpdate[exampleIndex] = 1.0 / rule.weightedError - 1.0;
				errorFlag = true;
			}
		}

		//update weights only if there is an error
		if (errorFlag)
		{
			weights = weights.cwiseProduct(weightUpdate);
			weights = weights / weights.sum();
			positiveTotalWeight = weights.head(nPositives).sum();
			negativeTotalWeight = 1 - positiveTotalWeight;

		}
	}
	void calcEmpiricalError(
		vector<StumpRule>* cascade
		, VectorXf& shifts
		, int layerCount
		, float& falsePositive
		, float& detectionRate

	) {
		int nFalsePositive = 0;
		int nFalseNegative = 0;

		if (TrainMode) {
			//initially let all be positive
			RowVectorXi verdicts, layerPrediction;
			verdicts.setOnes(sampleCount);
			layerPrediction.setZero(sampleCount);
			for (int layer = 0; layer < layerCount; layer++) {
				//set committee
				committee = cascade[layer];
				MatrixXd memberVerdict(committee.size(), sampleCount);
				RowVectorXd memberWeight(committee.size());
				//members, go ahead
				for (int member = 0; member < committee.size(); member++) {

					memberWeight[member] = log(1. / committee[member].weightedError - 1);
					int feature = committee[member].featureIndex;
					//#pragma omp parallel for schedule(static)
					for (int iterator = 0; iterator < sampleCount; iterator++) {
						int exampleIndex = getFeature_Index(feature, iterator).second;
						memberVerdict(member, exampleIndex) = (getFeature_Index(feature, iterator).first > committee[member].threshold ? 1 : -1) * committee[member].toggle + shifts[layer];
					}
				}
				//joint session
				if (committee.size() > 1) {
					RowVectorXd finalVerdict = memberWeight * memberVerdict;
					for (int exampleIndex = 0; exampleIndex < sampleCount; exampleIndex++)
						layerPrediction[exampleIndex] = finalVerdict[exampleIndex] > 0 ? 1 : -1;
				}
				else {
					for (int exampleIndex = 0; exampleIndex < sampleCount; exampleIndex++)
						layerPrediction[exampleIndex] = memberVerdict(0, exampleIndex) > 0 ? 1 : -1;
				}
				//those at -1, remain where you are
				verdicts = verdicts.cwiseMin(layerPrediction);
			}

			//evaluate prediction errors
			VectorXi agree = labels.cwiseProduct(verdicts.transpose());
			for (int exampleIndex = 0; exampleIndex < sampleCount; exampleIndex++) {
				if (agree[exampleIndex] < 0) {
					if (exampleIndex < nPositives) {
						nFalseNegative += 1;
					}
					else {
						nFalsePositive += 1;
					}
				}
			}
		}
		else {
			for (int exampleIndex = 0; exampleIndex < sampleCount; exampleIndex++) {
				bool hasFace;
				//look at this example's variance
				if (exampleIndex < 1000) {
					first = "vPos/vPos (";
					second = to_string(exampleIndex);
				}
				else
					first = "vNeg/vNeg (";
				second = to_string(negatives(exampleIndex - 1000));
				third = ").pgm";
				path = first + second + third;
				Mat img = imread(path, 0), mean, std_dev;

				meanStdDev(
					img,
					mean,
					std_dev
				);
				if (std_dev.at<double>(0, 0) < 1)
					hasFace = false;
				else
					hasFace = detect(img, cascade, shifts, round);
				if (!hasFace && exampleIndex < nPositives)
					nFalseNegative += 1;

				if (hasFace && exampleIndex >= nPositives)
					nFalsePositive += 1;

			}

		}
		//set the returned values
		falsePositive = nFalsePositive / (float)nNegatives;
		detectionRate = 1 - nFalseNegative / (float)nPositives;
	}
};

bool detect(Mat& img, vector<StumpRule>*& cascade, VectorXf& shifts, int round) {
	if (round == 0)
		return true;
	else {
		Mat integralImage;
		integral(img, integralImage);
		for (int r = 0; r < 25; r++)
			for (int c = 0; c < 25; c++)
				a[r][c] = integralImage.at<int>(r, c);
		for (int layer = 0; layer < round; layer++) {
			vector<StumpRule> committee = cascade[layer];
			double prediction = 0;
			int committeeSize = committee.size();
			for (int rule = 0; rule < committeeSize; rule++) {
				double featureValue = Compute_Feature(committee[rule].featureIndex);
				double vote = (featureValue > committee[rule].threshold ? 1 : -1) * committee[rule].toggle + shifts[layer];
				prediction += vote * log(1 / committee[rule].weightedError - 1);
			}
			if (prediction < 0)
				return false;
		}
		return true;
	}


}


int Compute_Feature(int z)
{
	int choice = Feature_Map::featureMap[z][0];
	int i = Feature_Map::featureMap[z][1], j = Feature_Map::featureMap[z][2];
	int w = Feature_Map::featureMap[z][3], h = Feature_Map::featureMap[z][4];
	switch (choice)
	{
	case 1:
		return sumint(i, j, w, h) - sumint(i + w, j, w, h);
	case 2:
		return sumint(i, j, w, h) - sumint(i + w, j, w, h) + sumint(i + 2 * w, j, w, h);
	case 3:
		return sumint(i, j, w, h) - sumint(i, j + h, w, h);
	case 4:
		return sumint(i, j, w, h) - sumint(i, j + h, w, h) + sumint(i, j + 2 * h, w, h);
	default:
		return sumint(i, j, w, h) + sumint(i + w, j + h, w, h) - sumint(i + w, j, w, h) - sumint(i, j + h, w, h);
	}
}

int main()
{


	vector<int> layerMemory;


	double accumulatedFalsePositive = 1;


	int nBoostingRounds = ceil(log(overallTargetFalsePositiveRate) / log(falsePositiveTarget)) + 20;

	vector<StumpRule>* cascade = new vector<StumpRule>[nBoostingRounds];
	VectorXf shifts;
	shifts.setZero(nBoostingRounds);




	for (int round = 0; round < nBoostingRounds && accumulatedFalsePositive > 1e-7; round++) {
		//start afresh
		TrainDataSet testSet(
			cascade				//cascade to get good negative examples
			, shifts
			, round				//layerCount in the cascade
			, false				//inTrain
		);

		TrainDataSet trainSet(
			cascade				//cascade to get good negative examples
			, shifts
			, round				//layerCount in the cascade
			, true				//inTrain
		);



		int committeeSizeGuide = min(20 + round * 10, 200);

		bool success = false;
		while (!success) {

			trainSet.adaboost();
			cascade[round] = trainSet.committee;
			bool overSized = (int)cascade[round].size() > committeeSizeGuide ? true : false;
			bool finalshift = overSized;

			//some parameters for shifting exception handling
			int shiftCounter = 0;
			Vector2i oscillationObserver;
			//shift to make do
			float shift = 0;
			//if finalshift, then try everything
			if (finalshift)
				shift = -1;
			float shiftUnit = 1e-2;
			float ctrlFalsePositive, ctrlDetectionRate, falsePositive, detectionRate;

			while (abs(shift) < 1.1) {
				shifts[round] = shift;

				//shift value depends on the worst detectionRate and falsePositive
				trainSet.calcEmpiricalError(cascade, shifts, round + 1, falsePositive, detectionRate);
				testSet.calcEmpiricalError(cascade, shifts, round + 1, ctrlFalsePositive, ctrlDetectionRate);
				float worstFalsePositive = max(falsePositive, ctrlFalsePositive);
				float worstDetectionRate = min(detectionRate, ctrlDetectionRate);

				//make sure that shifting always leads to a good detection rate
				//finalshift is enabled iff there's no viable shift to reach the targets
				if (finalshift) {
					if (worstDetectionRate >= 0.99) {
						cout << " final shift settles to " << shift << endl;
						break; //desperate break;
					}
					else {
						shift += 0.01;
						continue;
					}
				}


				if (worstDetectionRate >= detectionRateTarget && worstFalsePositive <= falsePositiveTarget) {
					success = true;
					break; //happy break
				}
				else if (worstDetectionRate >= detectionRateTarget && worstFalsePositive > falsePositiveTarget) {
					shift -= shiftUnit;
					shiftCounter++;
					oscillationObserver[shiftCounter % 2] = -1;
				}
				else if (worstDetectionRate < detectionRateTarget && worstFalsePositive <= falsePositiveTarget) {
					shift += shiftUnit;
					shiftCounter++;
					oscillationObserver[shiftCounter % 2] = 1;
				}
				else {
					finalshift = true;

					continue;
				}

				if (!finalshift && shiftCounter > 1 && oscillationObserver.sum() == 0) {

					shiftUnit /= 2;
					shift += oscillationObserver[shiftCounter % 2] == 1 ? -1 * shiftUnit : shiftUnit;

					if (shiftUnit < 1e-5) {
						finalshift = true;
					}
				}
			}

			if (overSized)
				break;
		}

		//record
		layerMemory.push_back(trainSet.committee.size());


		float detectionRate, falsePositive;
		trainSet.calcEmpiricalError(cascade, shifts, round + 1, falsePositive, detectionRate);
		testSet.calcEmpiricalError(cascade, shifts, round + 1, falsePositive, detectionRate);

		accumulatedFalsePositive *= falsePositive;
		cout << "Accumulated False Positive Rate is around " << accumulatedFalsePositive << endl;

		//set flags
		bool isFirst = round == 0 ? true : false;
		bool isLast = round == nBoostingRounds - 1 || accumulatedFalsePositive < 1e-7 ? true : false;


		//record the boosted rule into a target file

		ofstream output;
		output.open("Feature_Map.cpp", ios_base::app);

		int memberCount = trainSet.committee.size();
		//start a new item
		if (isFirst)
			output << "\ndouble Detector::stumps[][4]={\n";
		output.precision(10);
		output.setf(ios::fixed, ios::floatfield);
		for (int member = 0; member < memberCount; member++) {
			//a rule has 4 parameters
			output << "{" << trainSet.committee[member].featureIndex << "," << trainSet.committee[member].weightedError << "," << trainSet.committee[member].threshold << "," << trainSet.committee[member].toggle << "}";
			//handle the comma properly
			if (member == memberCount - 1 && isLast)
				output << "\n};\n";
			else
				output << ",\n";
		}
		//ok
		output.close();

		//release at the last round
		//if not, blackList is returned to remove examples
		if (isLast) {
			delete[] cascade;
			break;
		}
	}

	//record layerMemory
	ofstream output;
	output.open("Feature_Map.cpp", ios_base::app);
	//layerMemory size is the most reliable indicator of layers
	int layerCount = (int)layerMemory.size();
	output << "int Detector::layerCount=" << layerCount << ";\n";
	output << "int Detector::layerCommitteeSize[]={";
	for (int k = 0; k < layerCount; k++) {
		output << layerMemory[k];
		if (k < layerCount - 1)
			output << ",";
		else
			output << "};\n";
	}
	output << "float Detector::shifts[]={";
	output.precision(10);
	output.setf(ios::fixed, ios::floatfield);
	for (int k = 0; k < layerCount; k++) {
		output << shifts[k];
		if (k < layerCount - 1)
			output << ",";
		else
			output << "};\n";
	}
	output.close();
}
