#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


/////////////////  Draw Shape and Text  //////////////////////

void main() {

    // Blank Image
    Mat img(512, 512, CV_8UC3, Scalar(255,255,255));

    circle(img, Point(256, 256), 155, Scalar(255, 245, 53),-1);
    rectangle(img, Point(136, 226), Point(382, 286), Scalar(213, 132, 56), 3);
    line(img, Point(130, 296), Point(382, 296), Scalar(0, 0, 255), 2);

    putText(img, "Ajesh", Point(137, 262),FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(255, 128, 0), 3);
  
    imshow("Image", img);
   

    waitKey(0);

}

