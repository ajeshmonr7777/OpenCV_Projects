#include <iostream>
#include <cassert>
#include <opencv2/opencv.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>




void showImg(const std::string& name, cv::Mat& img) {
    cv::imshow(name, img);
    cv::waitKey(0);
}

void getBoundingBox(cv::Mat& img, cv::Rect& rect)
{

    cv::Point pt, pt2;
    int size;
    for (;;) {
        cv::Mat temp = img.clone();
        std::cout << "Insert x1,y1,x2,y2 Point:\n";
        std::cin >> pt.x >> pt.y >> pt2.x >> pt2.y;
        cv::rectangle(temp, pt, pt2, cv::Scalar(0, 255, 0), 3);
        showImg("boundingBox", temp);
        char choice;
        std::cout << "Do You want to change paramters (y/n)";
        std::cin >> choice;
        if (choice == 'n') {
            rect = cv::Rect(pt.x, pt.y, pt2.x, pt2.y);
            break;
        }
    }
}


int main() {
    string path = "Resources/test.png";

    cv::Mat src = cv::imread(path);
    assert(!src.empty());


    //    cv::Rect boundingBox(150,80,410,500); //My Image Bounding Box Value

        /* Reading Bounding Box */
    cv::Rect boundingBox;
    getBoundingBox(src, boundingBox);

    cv::Mat mask = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
    cv::Mat bgModel, fgModel;

    unsigned int iteration = 5; //Tune Parameter according to need
    cv::grabCut(src, mask, boundingBox, bgModel, fgModel, iteration, cv::GC_INIT_WITH_RECT);

    cv::Mat mask2 = (mask == 1) + (mask == 3);  // 0 = cv::GC_BGD, 1 = cv::GC_FGD, 2 = cv::PR_BGD, 3 = cv::GC_PR_FGD
    cv::Mat dest;
    src.copyTo(dest, mask2);

    showImg("dest", dest);
    cv::waitKey(0);
    return 0;

}