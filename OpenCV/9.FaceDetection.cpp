#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;


/////////////////  Images  //////////////////////

void main() {

    string path = "Resources/test.png";
    Mat img = imread(path);

    CascadeClassifier facecascade;
    facecascade.load("Resources/haarcascade_frontalface_default.xml");

    if (facecascade.empty()) {
        cout << "XML file not loaded" << endl;
    }

    vector <Rect> faces;
    facecascade.detectMultiScale(img, faces, 1.1, 10);

    for (int i = 0; i < faces.size(); i++) {
        rectangle(img, faces[i].tl(), faces[i].br(), Scalar(213, 132, 56), 3);
    }

    imshow("Image", img);
    waitKey(0);

}

