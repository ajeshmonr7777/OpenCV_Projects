#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


/////////////////  Images  //////////////////////

//void main() {
//
//    string path2 = "Resources/test.png";
//    string path1 = "Resources/cards.jpg";
//    Mat A = imread(path1);
//    Mat B = imread(path2);
//
//    double alpha = 0.7;
//
//    int min_x = 0 ;
//    int min_y = 0 ;
//
//    cv::Rect roi = cv::Rect(min_x, min_y, B.cols, B.rows);
//
//    // "out_image" is the output ; i.e. A with a part of it blended with B
//    cv::Mat out_image = A.clone();
//
//    // Set the ROIs for the selected sections of A and out_image (the same at the moment)
//    cv::Mat A_roi = A(roi);
//    cv::Mat out_image_roi = out_image(roi);
//
//    // Blend the ROI of A with B into the ROI of out_image
//    cv::addWeighted(A_roi, alpha, B, 1 - alpha, 0.0, out_image_roi);
//
//
//
//    imshow("Image1", A);
//    imshow("Image2", B);
//
//    imshow("Blended Image", out_image_roi);
//    waitKey(0);
//
//}



void main() {

    string path1 = "Resources/test.png";
    string path2 = "Resources/cards.jpg";
    Mat A = imread(path1);
    Mat B = imread(path2);

    double alpha = 0.8;
    double beta = 0.2;


    //int x = 0;
    //int y = 0;
    //int h = A.size().height ;
    //int w = A.size().width;

    int h = A.size().height;
    int w = A.size().width;
    int x = 10;
    int y = 20;

    Rect box (x, y, w, h);
    
    Mat roi1 = A;
    Mat roi2 = B(box);

    Mat frame;
    cvtColor(roi2, frame, COLOR_BGR2GRAY);


    Mat mask = (frame < 155);

    //cv::Mat roi1(A, cv::Rect(x, y, w, h));
    //cv::Mat roi2(B, cv::Rect(0, 0, w, h));



    Mat board(B.size().height, B.size().width, CV_8UC3, Scalar(255, 255, 255));

    A.copyTo(roi2, mask);


    cv::addWeighted(A, alpha, roi2, beta, 0.0, roi2);
    cv::namedWindow("Alpha Blend", 1);
    cv::imshow("Alpha Blend", B);
    imshow("mask", mask);
    imshow("Board", board);
    imshow("Copyto", roi2);

   
    cv::waitKey(0);



    /*imshow("Image1", A);
    imshow("Image2", B);*/

    //imshow("Blended Image", out_image_roi);
    /*waitKey(0);*/

}



