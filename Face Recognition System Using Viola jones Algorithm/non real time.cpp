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
#include <stdint.h>
#include "Feature_Map.h"

using namespace cv;
using namespace std;
using namespace Eigen;
typedef Matrix <long double, Dynamic, 1> VectorXld;
unsigned int a[25][25], counter = 0, neg_count = 0;

Mat image, image2, Big_image;

float scale_x, scale_y;

int size_;

int sumint(int z_1, int z_2, int z_3, int z_4)
{
	return (a[z_2 + z_4 - 1][z_1 + z_3 - 1] + a[z_2 - 1][z_1 - 1] - a[z_2 + z_4 - 1][z_1 - 1] - a[z_2 - 1][z_1 + z_3 - 1]);
}


struct window {
	int pos_i;
	int pos_j;
	int side;
};
struct stumpRule
{
	int featureIndex;
	double threshold;
	double margin;
	double weightedError;
	int toggle;
};

double computeFeature(int z, Mat& intImg)
{
	for (int r = 0; r < 25; r++) {
		for (int c = 0; c < 25; c++) {
			a[r][c] = intImg.at<int>(r, c);
		}
	}
	int choice = Feature_Map::featureMap[z][0];
	int i = Feature_Map::featureMap[z][1], j = Feature_Map::featureMap[z][2];
	int w = Feature_Map::featureMap[z][3], h = Feature_Map::featureMap[z][4];
	switch (choice)
	{
	case 1:
	{
		return sumint(i, j, w, h) - sumint(i + w, j, w, h);
		break;
	}
	case 2:
	{
		return sumint(i, j, w, h) - sumint(i + w, j, w, h) + sumint(i + 2 * w, j, w, h);
		break;
	}
	case 3:
	{
		return sumint(i, j, w, h) - sumint(i, j + h, w, h);
		break;
	}
	case 4:
	{
		return sumint(i, j, w, h) - sumint(i, j + h, w, h) + sumint(i, j + 2 * h, w, h);
		break;
	}
	default:
		return sumint(i, j, w, h) + sumint(i + w, j + h, w, h) - sumint(i + w, j, w, h) - sumint(i, j + h, w, h);
	}
}
bool detectFace(
	Mat& integralImage
	, double varianceNormalizer
	, VectorXf& tweaks
	, vector<stumpRule> const* cascade
	, int defaultLayerNumber
) {
	//everything is a face if no layer is involved

	if (defaultLayerNumber == 0)
		return true;
	//now to the cascade: you may choose the number of layers used in the detection
	int layerCount = defaultLayerNumber < 0 ? tweaks.size() : defaultLayerNumber;
	double prediction = 0.0;
	neg_count = 0;
	for (int layer = 0; layer < layerCount; layer++)
	{
		vector<stumpRule> committee = cascade[layer];
		int committeeSize = committee.size();
		for (int rule = 0; rule < committeeSize; rule++)
		{
			double featureValue = computeFeature(committee[rule].featureIndex, integralImage) / varianceNormalizer;
			//cout << committee[rule].threshold<<", "<< featureValue << endl;

			double vote = (featureValue > committee[rule].threshold ? 1 : -1) * committee[rule].toggle + tweaks[layer];
			//cout << featureValue - committee[rule].threshold << endl;
			if (committee[rule].weightedError == 0)
			{
				//very well then
				if (rule == 0)
					return vote > 0 ? true : false;
				else
					cout << "Find an invalid rule.";
			}
			//no 0.5 since only sign matters
			prediction += vote * log(1 / committee[rule].weightedError - 1);
		}

		if (prediction < 0)
		{
			return false;
			//neg_count++;
			//if (3 == neg_count) return false;
		}
	}
	return true;
}

void tscan(
	Mat& img
	, int& nRows
	, int& nCols
	, int defaultLayerNumber
	, vector<stumpRule>* cascade
	, VectorXf& tweaks
	, vector<window>& toMark
) {
	//build integral image and windowd integral image
	Mat part, mean, stddev, resized, intImg;

	nRows = img.size().width; // 512
	nCols = img.size().height; //512


	//get down to the business
	int sampleSize = 24;
	int possibleULCorners = (nRows - sampleSize + 1) * (nCols - sampleSize + 1); //239121
//#pragma omp parallel for schedule (static)
	for (int ij = 0; ij <= possibleULCorners; ij = ij + 2)
	{
		//#pragma omp critical
		int i = ij / (nCols - sampleSize + 1);
		int j = ij % (nCols - sampleSize + 1);
		double scale = 1;
		window area;
		area.pos_i = i;
		area.pos_j = j;
		area.side = sampleSize;

		//multiple scale detection
		while (i + area.side <= nRows && j + area.side <= nCols)
		{
			part = img(Rect(area.pos_i, area.pos_j, area.side, area.side));
			meanStdDev(part, mean, stddev);
			double std = stddev.at<double>(0, 0);

			if (!(isnan(std) || std < 1)) {
				if (area.side > 24)
					resize(img, part, Size(24, 24), INTER_LINEAR);
				integral(part, intImg);
				if (detectFace(intImg, std / 1e4, tweaks, cascade, defaultLayerNumber))
				{
					//#pragma omp critical
					{

						toMark.push_back(area);
					}
				}
			}
			scale *= 1.5;
			area.side = sampleSize * scale + 1;
		}
	}
}
void scan()
{
	int defaultLayerNumber = -1;
	float required_nFriends = 3.0;
	//Mat frame, img;
	//VideoCapture cap;
	//int deviceID = 0; // 0 = open default camera
	//int apiID = cv::CAP_ANY; // 0 = autodetect default API
	//cap.open(deviceID, apiID);

	//read in cascade
	vector<stumpRule>* cascade = NULL;
	int layerCount = 13;
	cascade = new vector<stumpRule>[layerCount];
	VectorXf tweaks;
	tweaks.setZero(layerCount);
	int linearCounter = 0;
	//this should be done in a linear fashion
	for (int layer = 0; layer < layerCount; layer++)
	{
		int committeeSize = Feature_Map::layerCommitteeSize[layer];

		tweaks[layer] = Feature_Map::shifts[layer];
		for (int member = 0; member < committeeSize; member++)
		{
			double* rule = Feature_Map::stumps[linearCounter];
			//set a new stump
			stumpRule newcomer;
			newcomer.featureIndex = rule[0];
			newcomer.weightedError = rule[1];
			newcomer.threshold = rule[2];
			newcomer.toggle = rule[3];
			cascade[layer].push_back(newcomer);
			linearCounter++;
		}
	}

	Mat img = imread("0.jpg", 0);
	scale_x = img.size().width / (39 * 3);
	scale_y = img.size().height / (29 * 3);
	scale_x = 400 / 39;
	scale_y = 300 / 29;
	resize(img, Big_image, Size(scale_x * 39, scale_y * 29), INTER_LINEAR);
	//resize(imgage, Big_image, Size(scale_x, scale_y), INTER_LINEAR);
	resize(Big_image, img, Size(39, 29), INTER_LINEAR);

	vector<window> toMark, combined;

	//imshow("Live", frame);
	//if (waitKey(5) >= 0)
		//break;

	//scan the file
	int nRows, nCols;
	tscan(img, nRows, nCols, defaultLayerNumber, cascade, tweaks, toMark);
	/*float theta = 5 / 180. * M_PI;
	int center_i, center_j;
	for (int bf = 1; bf < 3; bf++) {
		float curTheta = pow(-1., bf) * theta;
		rotateImage(img, "rotated.png", curTheta, center_i, center_j);
		tscan("rotated.png", nRows, nCols, defaultLayerNumber, cascade, tweaks, toMark);
		int num = toMark.size();
		for (int k = 0; k < num; k++)
			rotateCoordinate(toMark[k].pos_i, toMark[k].pos_j, center_i, center_j, curTheta, toMark[k].pos_i, toMark[k].pos_j);
		combined.reserve(combined.size() + toMark.size());
		combined.insert(combined.end(), toMark.begin(), toMark.end());
		toMark.resize(0);
	}*/
	delete[] cascade;
	//due to rotation, the coordinates might not be legal
	/*for (int i = 0; i < (int)combined.size(); i++)
		if (isLegal(combined[i], nRows, nCols))
			toMark.push_back(combined[i]);*/
			//four modes of post-processing
	int thickness = 2;
	float avg_x = 0.0, avg_y = 0.0, avg_side = 0.0;
	size_ = toMark.size();

	if (10 < size_)
	{
		for (int k = 0; k < size_; k++) {
			// Top Left Corner
			avg_x += toMark[k].pos_i;
			avg_y += toMark[k].pos_j;
			avg_side += toMark[k].side;
		}
		avg_x /= size_;
		avg_y /= size_;
		avg_side /= size_;

		Point p1(avg_x * scale_x * 1.1, avg_y * scale_y * 1.1);
		Point p2((avg_x + avg_side) * scale_x * 0.9, (avg_y + avg_side) * scale_y * 0.9);
		rectangle(Big_image, p1, p2, Scalar(255, 0, 0), thickness, LINE_8);
		image2 = Big_image(Rect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y));
		imshow("Out", image2);
		imwrite("D:/Face Recognition/0.jpg", image2);
	}

	imshow("Output", Big_image);
	waitKey(0);
	/*for (int ppMode = 0; ppMode < 4; ppMode++) {
		combined = toMark;
		highlight(file, combined, ppMode, required_nFriends);*/
		//}
}

int main()
{

	//224
	//while(1)
	scan();

}