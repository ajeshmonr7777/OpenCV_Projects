#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


/////////////////  Resize and Croped  //////////////////////

void main() {

    string path = "Resources/test.png";
    Mat img = imread(path);
    int w, h, cX, cY;

    Mat imgResize, imgCrop;
    Mat Rotation;

    cout << img.size() << endl;
    cout << "Width " << img.size().width << endl;   
    cout << "Height " << img.rows << endl;            // Differrent way

    w = img.size().width;
    h = img.rows;

    cX = (int)w / 2;
    cY = (int)h / 2;
    cout << "(Cx, Cy) = (" << cX << " , " << cY << ")" << endl;

    // cropping Top Left
    Rect top_left_roi(0, 0, cX, cY);
    Mat top_left;
    top_left = img(top_left_roi);




    
    // resize(img, imgResize, Size(640,480));
    resize(img, imgResize, Size(),0.5,0.5);

    // Cropping randomly
    Rect roi(100, 100, 300, 250);

    imgCrop = img(roi);

    // Flip Image
    Mat flip_horizontal;
    flip(img, flip_horizontal,-1);


    // Translation
    Mat shift_img;
    float data[6] = { 1,0,30,0,1,50 };
    Mat shift_matrix_float = Mat(2, 3, CV_32F, data);
    cout << shift_matrix_float;

    
    Mat shift_matrix = Mat(2, 3, CV_64F);   // Convert CV_64F Format
    shift_matrix_float.convertTo(shift_matrix, CV_64F);

    warpAffine(img, shift_img, shift_matrix, img.size());

    // Rotation 
    Mat rotate_matrix, rotated;
    Point2f rotation_center(w / 2, h / 2);

    rotate_matrix = getRotationMatrix2D(rotation_center, 45, 1.0);
    warpAffine(img, rotated, rotate_matrix, img.size());






    imshow("Image", img);
    imshow("Image_Resized", imgResize);
    imshow("Image Croped", imgCrop);
    imshow("TopLeft Cropp", top_left);
    imshow("Flip Horizontal", flip_horizontal);
    imshow("Translate Flip", shift_img);
    imshow("Rotated by 45 Degrees", rotated);
    
    waitKey(0);

}

