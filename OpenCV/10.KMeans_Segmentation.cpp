#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <iostream>

using namespace cv;
using namespace std;


/////////////////  KMeans  //////////////////////



void main() {

    string path = "Resources/cards.jpg";
    Mat ocv = imread(path);

    Mat data;
    ocv.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());

    //do KMeans
    Mat labels, centers;
    kmeans(data, 8, labels, TermCriteria(, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

    //reshape both to a single row of Vec3f pixels: 
    centers = centers.reshape(3, centers.rows);
    data = data.reshape(3, data.rows);

    // replace pixel values with their centre value:

    Vec3f* p = data.ptr<Vec3f>();
    for (size_t i = 0; i < data.rows; i++) {
        int center_id = labels.at<int>(i);
        p[i] = centers.at<Vec3f>(center_id);
    }

    // Back to 2d, and uchar:

    ocv = data.reshape(3, ocv.rows);
    ocv.convertTo(ocv, CV_8U);

    imshow("KMeans", data);
    imshow("Orginal Image", ocv);


    waitKey(0);

}