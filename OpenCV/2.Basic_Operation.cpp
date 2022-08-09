#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;


/////////////////  Basic Functions  //////////////////////

void main() {

    string path = "Resources/test.png";
    Mat img = imread(path);
    Mat imageGray,imageBlur,imageCanny, imageDia, imageErode, sum_rgb;

    

    // For color image i.e. 3 channel
    Vec3b intensity = img.at<Vec3b>(10, 10);
    cout << "BGR " << intensity << "\n";


    // Print Individual components
    int blue = intensity.val[0];
    cout << "Blue" << blue << "\n";
    int green = intensity.val[1];
    cout << "Green" << green << "\n";
    int red = intensity.val[2];
    cout << "Red" << red << "\n";

    // 3 channel to store B,G,R
    Mat rgbchannel[3];

    // split image
    split(img, rgbchannel);

    //Plot Individual Components 
    namedWindow("Blue", WINDOW_AUTOSIZE);
    imshow("Red", rgbchannel[0]);

    namedWindow("Green", WINDOW_AUTOSIZE);
    imshow("Green", rgbchannel[1]);

    namedWindow("Red", WINDOW_AUTOSIZE);
    imshow("Blue", rgbchannel[2]);

    // merge : (input, num_of_channel, output)
    merge(rgbchannel, 3, sum_rgb);
    imshow("merged", sum_rgb);

    cvtColor(img, imageGray, COLOR_BGR2GRAY);
    GaussianBlur(img, imageBlur,Size(3,3),5,0); // Source and destination give inside the parameter
    Canny(imageBlur, imageCanny, 50, 100);   

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    
    

    dilate(imageCanny, imageDia, kernel);
    erode(imageDia, imageErode, kernel);



    imshow("Image", img);
    imshow("Image Gray", imageGray);
    imshow("Image Blur", imageBlur);
    imshow("Image Canny", imageCanny);
    imshow("Image Dilation", imageDia);
    imshow("Image Erosion", imageErode);
    waitKey(0);

}

