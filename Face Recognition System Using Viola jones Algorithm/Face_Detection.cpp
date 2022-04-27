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

using namespace cv;
using namespace std;
using namespace Eigen;

typedef Matrix <long double, Dynamic, 1> VectorXld;
int a[25][25], counter=0;
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
int sumint(int z_1, int z_2, int z_3, int z_4)
{
	//int_img.at<int>(r, c);
	return (a[z_2 + z_4 - 1][z_1 + z_3 - 1] + a[z_2 - 1][z_1 - 1] - a[z_2 + z_4 - 1][z_1 - 1] - a[z_2 - 1][z_1 + z_3 - 1]);
}
int computeFeature(int z, Mat& int_img)
{

	for (int r = 0; r < 25; r++) {
		for (int c = 0; c < 25; c++) {
			a[r][c] = int_img.at<int>(r, c);
			//cout << a[r][c] << " ";
		}
		//cout << endl;
	}


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
	for (int layer = 0; layer < layerCount ; layer++)
	{
		vector<stumpRule> committee = cascade[layer];
		int committeeSize = committee.size();
		for (int rule = 0; rule < committeeSize; rule++)
		{

			double featureValue = computeFeature(committee[rule].featureIndex, integralImage) / varianceNormalizer;
			//cout << committee[rule].threshold<<", "<< featureValue << endl;

			double vote = (featureValue > committee[rule].threshold ? 1 : -1) * committee[rule].toggle + tweaks[layer];
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
		//cout << prediction << endl;
		if (prediction < 0)
			return false;
	}
	counter++;
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

	//nRows = 75; // 512
	//nCols = 75; //512

	//get down to the business
	int sampleSize = 24;
	int possibleULCorners = (nRows - sampleSize + 1) * (nCols - sampleSize + 1); //239121
//#pragma omp parallel for schedule (static)
	for (int ij = 0; ij <= possibleULCorners; ij++) {
		//#pragma omp critical
		int i = ij / (nCols - sampleSize + 1);
		int j = ij % (nCols - sampleSize + 1);
		double scale = 1;
		window area;
		area.pos_i = i;
		area.pos_j = j;
		area.side = sampleSize;

		//multiple scale detection
		while (i + area.side <= nRows && j + area.side <= nCols) {

			part = img(Rect(area.pos_i, area.pos_j, area.side, area.side));
			meanStdDev(part, mean, stddev);
			double std = stddev.at<double>(0, 0);

			if (!(isnan(std) || std < 1)) {
				if (area.side > 24)
					resize(img, part, Size(24, 24), INTER_LINEAR);
				integral(part, intImg);
				if (detectFace(intImg, std / 1e4, tweaks, cascade, defaultLayerNumber)) {
					//#pragma omp critical
										//{
					//toMark.push_back(area);
					//cout << area.pos_i << ", " << area.pos_j << endl;
					//}
				}
			}
			//next scale
			scale *= 1.5;//1.5
			area.side = sampleSize * scale + 0.5;
		}
	}
}
void scan(
	Mat& img
	, int defaultLayerNumber
	, float required_nFriends
) {
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

	vector<window> toMark, combined;


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
	//for (int k = 0; k < toMark.size(); k++) {
		// Top Left Corner
		//Point p1(toMark[k].pos_i, toMark[k].pos_j);

		// Bottom Right Corner
		//Point p2(toMark[k].pos_i + toMark[k].side, toMark[k].pos_j + toMark[k].side);



		// Drawing the Rectangle
		//rectangle(img, p1, p2,Scalar(255, 0, 0),thickness, LINE_8);

		// Show our image inside a window

	//}
	//imshow("Output", img);
	//waitKey(0);
	/*for (int ppMode = 0; ppMode < 4; ppMode++) {
		combined = toMark;
		highlight(file, combined, ppMode, required_nFriends);*/
		//}
}
//void rotateCoordinate(
//	int input_i
//	, int input_j
//	, int center_i
//	, int center_j
//	, float theta
//	, int& output_i
//	, int& output_j
//) {
//	MatrixXf rotMatrix(2, 2);
//	rotMatrix << cos(theta), sin(theta),
//		-1 * sin(theta), cos(theta);
//	Vector2f coordinate;
//	coordinate << input_j - center_j, input_i - center_i;
//	coordinate = rotMatrix * coordinate;
//	output_i = coordinate(1) + center_i;
//	output_j = coordinate(0) + center_j;
//}
//
//void rotateImage(
//	Mat &image
//	, const char* outfile
//	, float theta
//	, int& center_i
//	, int& center_j
//) {
//	int nRows, nCols, nChannels;
//	nRows = image.size().width;
//	nCols = image.size().height;
//	int imsize = nRows * nCols;
//	MatrixXf* rotated = new MatrixXf[1];
//	rotated[0].setZero(nRows, nCols);
//	center_i = nRows / 2;
//	center_j = nCols / 2;
//#pragma omp parallel for schedule(static)
//	for (int ij = 0; ij < imsize; ij++) {
//		int i = ij / nCols;
//		int j = ij % nCols;
//		int si, sj;
//		rotateCoordinate(i, j, center_i, center_j, theta, si, sj);
//		if (!(i < 0 || i > nRows - 1 || j < 0 || j > nCols - 1 || si < 0 || si > nRows - 1 || sj < 0 || sj > nCols - 1))
//			rotated[0](i, j) = image[0](si, sj);
//	}
//	//write out
//	imwrite(outfile, rotated, 1);
//	delete[] image;
//}
//bool isLegal(
//	window& area
//	, int nRows
//	, int nCols
//) {
//	int i = area.pos_i;
//	int j = area.pos_j;
//	if (i < 0 || i > nRows - 1 || j < 0 || j > nCols - 1)
//		return false;
//	i += area.side - 1;
//	j += area.side - 1;
//	if (i < 0 || i > nRows - 1 || j < 0 || j > nCols - 1)
//		return false;
//	else
//		return true;
//}



//highlight a rectangle part of an image
//void highlight(
//	const char* inputName
//	, vector<window>& areas
//	, int ppMode
//	, float required_nFriends
//) {
//	int nRows, nCols;
//	bool isColor;
//	MatrixXf* original = convertToColor(inputName, nRows, nCols, isColor);
//	int nwindows = areas.size();
//	//no detection, no post-processing
//	ppMode = nwindows == 0 ? 0 : ppMode;
//	augmentedPostProcessing(original, nRows, nCols, isColor, required_nFriends, areas, ppMode);
//	nwindows = areas.size();
//	for (int k = 0; k < nwindows; k++) {
//
//		//take the parameters
//		int cornerI = areas[k].pos_i;
//		int cornerJ = areas[k].pos_j;
//		int side = areas[k].side;
//
//		//I'm working with a color image, and always highlight with green
//		for (int i = cornerI; i < cornerI + side; i++)
//			for (int j = cornerJ; j < cornerJ + side; j++) {
//				bool paint = false;
//				if (abs(cornerI + PEN_WIDTH / 2 - i) <= PEN_WIDTH / 2 && abs(cornerJ + side / 2 - j) <= side / 2)
//					paint = true;
//				else if (abs(cornerI + side - 1 - PEN_WIDTH / 2 - i) <= PEN_WIDTH / 2 && abs(cornerJ + side / 2 - j) <= side / 2)
//					paint = true;
//				else if (abs(cornerJ + PEN_WIDTH / 2 - j) <= PEN_WIDTH / 2 && abs(cornerI + side / 2 - i) <= side / 2)
//					paint = true;
//				else if (abs(cornerJ + side - 1 - PEN_WIDTH / 2 - j) <= PEN_WIDTH / 2 && abs(cornerI + side / 2 - i) <= side / 2)
//					paint = true;
//				if (paint) {
//					original[0](i, j) = 0;
//					original[1](i, j) = 255;
//					original[2](i, j) = 0;
//				}
//			}
//	}
//
//	//output
//	if (ppMode == 0)
//		imwrite("detectedraw.png", original, 3);
//	else if (ppMode == 1)
//		imwrite("ppRobust.png", original, 3);
//	else if (ppMode == 2)
//		imwrite("ppSkin.png", original, 3);
//	else
//		imwrite("ppBoth.png", original, 3);
//}



int main()
{
	int defaultLayerNumber = -1;
	float required_nFriends = 3;
	Mat image;
	int z = 0;
	for (z = 0; z < 15520; z++)
	{
		//image = imread("neg/neg (" + to_string(z) + ").pgm", 0);
		image = imread("pos/" + to_string(z) + ".pgm", 0);
		scan(image, defaultLayerNumber, required_nFriends);
	}
	cout << endl << endl << " efficiency = "<<counter << " / " << z;
}
