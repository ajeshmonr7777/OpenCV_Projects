#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


/////////////////  PROJECT 2  //////////////////////

Mat imgOrginal, imgGray,imgBlur, imgCanny, imgThre, imgDia, imageErode, imgWarp, imgCrop,imgGr, imgBinThr, imgAdaThr;
vector <Point> initialPoints,docPoints;

float w = 420, h = 596;

Mat preProcessing(Mat img) 
{
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(img, imgBlur, Size(3, 3), 5, 0); // Source and destination give inside the parameter
    Canny(imgBlur, imgCanny, 50, 100);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(imgCanny, imgDia, kernel);
    // erode(imgDia, imageErode, kernel);
    return imgDia;

}



vector <Point> getContours(Mat imgDia) {

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;



    findContours(imgDia, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //drawContours(img, contours, -1, Scalar(255, 0, 255), 2);

    vector<Rect> boundRect(contours.size());
    vector<vector<Point>> conPoly(contours.size());

    vector<Point> biggest;
    int maxArea =0;

    for (int i = 0; i < contours.size(); i++) {
        int area = contourArea(contours[i]);
        cout << area << endl;


        string ObjectType;
        if (area > 1000) {
            float peri = arcLength(contours[i], true);
            approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);

            if (area > maxArea && conPoly[i].size()==4) {
                biggest = { conPoly[i][0],conPoly[i][1],conPoly[i][2],conPoly[i][3] };
                //drawContours(imgOrginal, conPoly, i, Scalar(255, 0, 255), 2);
                maxArea = area;
            }
          



            //drawContours(imgOrginal, conPoly, i, Scalar(255, 0, 255), 2);
            // rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(213, 132, 56), 3);


        }
    }
    return biggest;
}

void drawPoints(vector<Point> points, Scalar color) {

    for (int i = 0; i < points.size(); i++) {
        circle(imgOrginal, points[i], 10, color, -1); 
        putText(imgOrginal, to_string(i), points[i], FONT_HERSHEY_PLAIN, 2, color, 2);
     }
}



vector<Point> reorder(vector<Point> points)
{
    vector<Point> newPoints;
    vector<int> sumPoints, subPoints;

    for (int i = 0;i < 4;i++) {
        sumPoints.push_back(points[i].x + points[i].y);
        subPoints.push_back(points[i].x - points[i].y);
    }
    
    newPoints.push_back(points[min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]);
    newPoints.push_back(points[max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]);
    newPoints.push_back(points[min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]);
    newPoints.push_back(points[max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]);

    return newPoints;
}

Mat getWarp(Mat img, vector<Point> points, float w, float h)
{
    Point2f src[4] = { points[0], points[1],points[2], points[3] };
    Point2f dst[4] = { {0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h} };

    Mat matrix = getPerspectiveTransform(src, dst);
    warpPerspective(img, imgWarp, matrix, Point(w, h));

    return imgWarp;
}



void main() {

    string path = "Resources/paper.jpg";
    imgOrginal = imread(path);

    
   
 
    //resize(imgOrginal, imgOrginal, Size(), 0.5, 0.5);

    // PreProcessing
    imgThre = preProcessing(imgOrginal);





    // GetContours - Biggest
    initialPoints = getContours(imgThre);
    //drawPoints(initialPoints, Scalar(0, 0, 255));
    docPoints = reorder(initialPoints);
    //drawPoints(docPoints, Scalar(0, 255, 255));



    // Warp
    imgWarp = getWarp(imgOrginal, docPoints, w, h);


    //Crop
    int Cropval = 5;
    Rect roi(Cropval, Cropval, w - (2 * 5), h - (2 * 5));
    imgCrop = imgWarp(roi);
    cout << roi;

    // Thresholding 
    cvtColor(imgCrop, imgGr, COLOR_BGR2GRAY);
    threshold(imgGr, imgBinThr, 125, 255, THRESH_BINARY);
    adaptiveThreshold(imgGr, imgAdaThr, 125,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,5,15);
    


    imshow("Image", imgOrginal);
    //imshow("Image Threshold", imgThre);
    //imshow("Image Warp", imgWarp);
    imshow("Image Crop", imgCrop);
    imshow("Image Bianry Threshold ", imgBinThr);
    imshow("Image Adaptive Threshold ", imgAdaThr);
    waitKey(0);

    
    

}