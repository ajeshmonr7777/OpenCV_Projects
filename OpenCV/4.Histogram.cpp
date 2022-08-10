#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;


/////////////////  Draw Shape and Text  //////////////////////

void main() {

    string path = "Resources/cards.jpg";
    Mat img = imread(path);

    cvtColor(img, img, COLOR_BGR2BGRA);

   /* MatND histogram;
    int histSize = 256;
    const int* channel_numbers = { 0 };
    float channel_range[] = { 0.0,256.0 };
    const float* channel_ranges = channel_range;
    int number_bins = histSize;

    calcHist(&img, 1, 0, Mat(), histogram, 1, &number_bins, &channel_ranges);*/


    Mat histEqualized;
    equalizeHist(img, histEqualized);

    imshow("HistImage", histEqualized);
    imshow("Image", img);
    


    waitKey();

}

