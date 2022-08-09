#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


/////////////////  Contour Detection  //////////////////////
Mat imageGray, imageBlur, imageCanny, imageDia, imageErode;

void getContours(Mat imgDia, Mat img) {

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(imgDia, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //drawContours(img, contours, -1, Scalar(255, 0, 255), 2);

    vector<Rect> boundRect(contours.size());
    
    for (int i = 0; i < contours.size(); i++) {
        int area = contourArea(contours[i]);
        cout << area << endl;
        vector<vector<Point>> conPoly(contours.size());

        string ObjectType;
        if (area > 1000) {
            float peri = arcLength(contours[i], true);
            approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
            
            cout << conPoly[i].size() << endl;
            boundRect[i] = boundingRect(conPoly[i]);
            

            int objCor = (int)conPoly[i].size();

            if (objCor == 3) { ObjectType = "Tri"; }
            if (objCor == 4) {

                float aspRatio = (float)boundRect[i].width / (float)boundRect[i].height;
                cout << aspRatio << endl;
                if (aspRatio > 0.95 && aspRatio < 1.05) { ObjectType = "Square"; }
                else {
                    ObjectType = "Rect";
                }
            }
                
            if (objCor > 4) { ObjectType = "Circle"; }

            rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(213, 132, 56), 3);
            drawContours(img, conPoly, i, Scalar(255, 0, 255), 2);

            putText(img, ObjectType,{ boundRect[i].x, boundRect[i].y - 5}, FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 128, 0), 1);
        }
    }
}

    void main() {

    string path = "Resources/shapes.png";
    Mat img = imread(path);


    //Preprocessing
    cvtColor(img, imageGray, COLOR_BGR2GRAY);
    GaussianBlur(img, imageBlur, Size(3, 3), 5, 0); // Source and destination give inside the parameter
    Canny(imageBlur, imageCanny, 50, 150);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(imageCanny, imageDia, kernel);

    getContours(imageDia, img);

    imshow("Image", img);
    //imshow("Image Gray", imageGray);
    //imshow("Image Blur", imageBlur);
    //imshow("Image Canny", imageCanny);
    //imshow("Image Dilation", imageDia);
    waitKey(0);

}
