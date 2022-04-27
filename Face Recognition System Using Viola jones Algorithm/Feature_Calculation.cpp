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
#include <iomanip>

using namespace cv;
using namespace std;

int a[25][25];//defalt frame size r = 480, c = 640 , mul factor = 0.9
int tem, r, c, w, h, nRows = 24, nCols = 24, feature, pic_count = 0, feature_count, length_count;
float sd, m;
string path, first, second, third;

vector <pair <int, int>> Feature_vector;

int sumint(int z_1, int z_2, int z_3, int z_4)
{
    return (a[z_2 + z_4 - 1][z_1 + z_3 - 1] + a[z_2 - 1][z_1 - 1] - a[z_2 + z_4 - 1][z_1 - 1] - a[z_2 - 1][z_1 + z_3 - 1]);
}

int main()
{
    //ofstream File_2("Featute_Map.cpp");
    //File_2 << "#include " << "Featute_Map.h" << endl;
    //File_2 << "int Feature_Map::featureMap[][5] = {" << endl;
    //featureMap[][5] = {a/b/c/d/e,i,j,w,h}
    //featureMap[][5] = {1/2/3/4/5,i,j,w,h}

    pic_count = 0;
    while (pic_count < 3019)//For +ve image put < 1000
    {
        first = "d:/neg ("; //For +ve image use "d:/"
        second = to_string(pic_count);
        third = ").pgm";

        path = first + second + third;

        second = to_string(pic_count+1000);//For +ve image comment this line

        ofstream File_1(second + ".txt");

        Mat img = imread(path, 0), int_img;

        integral(img, int_img);

        for (r = 1; r < 25; r++)
            for (c = 1; c < 25; c++)
            {
                a[r][c] = int_img.at<int>(r, c);
            }

        for (int part = 2; part < 4; part++)
        {
            int temp = part > 2 ? 1 : 0;
            for (int i = 1; i <= nRows; i++)
                for (int j = 1; j <= nCols; j++)
                {
                    for (h = 1; j + h - 1 <= 24; h++)
                        for (w = 1; i + (2 + temp) * w - 1 <= 24; w++)
                        {//a
                            feature = sumint(i, j, w, h) - sumint(i + w, j, w, h);

                            if (part > 2)
                            {//b
                                feature = feature + sumint(i + 2 * w, j, w, h);

                            }
                            File_1 << setw(6) << feature << endl;
                            //if (part > 2)
                            //{
                                //File_2 << "{" << 2 << "," << i << "," << j << "," << w << "," << h << "}," << endl;
                            //}
                            //else
                                //File_2 << "{" << 1 << "," << i << "," << j << "," << w << "," << h << "}," << endl;
                        }
                }
        }

        for (int part = 2; part < 4; part++)
        {
            int temp = part > 2 ? 1 : 0;
            for (int i = 1; i <= nRows; i++)
                for (int j = 1; j <= nCols; j++)
                {
                    for (h = 1; j + (2 + temp) * h - 1 <= 24; h++)
                        for (w = 1; i + w - 1 <= 24; w++)
                        {//c
                            feature = sumint(i, j, w, h) - sumint(i, j + h, w, h);

                            if (part > 2)
                            {//d
                                feature = feature + sumint(i, j + 2 * h, w, h);
                            }
                            File_1 << setw(6) << feature << endl;
                            //if (part > 2)
                             //{
                                  //File_2 << "{" << 4 << "," << i << "," << j << "," << w << "," << h << "}," << endl;
                             //}
                             //else
                                  //File_2 << "{" << 3 << "," << i << "," << j << "," << w << "," << h << "}," << endl;

                        }
                }
        }

        for (int i = 1; i <= nRows; i++)
            for (int j = 1; j <= nCols; j++)
            {
                for (h = 1; j + 2 * h - 1 <= 24; h++)
                    for (w = 1; i + 2 * w - 1 <= 24; w++)
                    {
                        feature = sumint(i, j, w, h) + sumint(i + w, j + h, w, h) - sumint(i + w, j, w, h) - sumint(i, j + h, w, h);
                        File_1 << setw(6) << feature << endl;

                        //if (i == 23 && j == 23)
                            //File_2 << "{" << 5 << "," << i << "," << j << "," << w << "," << h << "}" << endl << "}";
                        //else
                            //File_2 << "{" << 5 << "," << i << "," << j << "," << w << "," << h << "}," << endl;
                    }
            }

        File_1.close();
        //File_2.close();
        pic_count++;
        //}
        //waitKey(0);
        //destroyAllWindows();
    }
    return 0;
}

