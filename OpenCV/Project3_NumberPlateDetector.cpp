#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;


/////////////////  PROJECT 3  //////////////////////








void main() {

    VideoCapture cap(0);

    
    Mat img;
    

    CascadeClassifier platecascade;
    platecascade.load("Resources/haarcascade_russian_plate_number.xml");

    if (platecascade.empty()) {
        cout << "XML file not loaded" << endl;
    }

    vector <Rect> plates;

    while (true) {
        cap.read(img);
        platecascade.detectMultiScale(img, plates, 1.1, 10);

        for (int i = 0; i < plates.size(); i++) {
            Mat imgCrop = img(plates[i]);
            imshow(to_string(i), imgCrop);
            imwrite("Resources/Plates/" + to_string(i) + ".png",imgCrop);

            rectangle(img, plates[i].tl(), plates[i].br(), Scalar(213, 132, 56), 3);
        }

        imshow("Image", img);
        waitKey(1);
    }

}





