#include <iostream>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <string.h>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

using namespace cv;
using namespace std;
using namespace dnn;


void showImg(const std::string& name, cv::Mat& img) {
    cv::imshow(name, img);
    cv::waitKey(0);
}

void getBoundingBox(cv::Mat& img, cv::Rect& rect)
{
    std::vector<std::string> class_names;
    ifstream ifs(string("Resources/input/object_detection_classes_coco.txt").c_str());
    string line;
    while (getline(ifs, line))
    {
        class_names.push_back(line);
    }

    // load the neural network model
    auto model = readNet("Resources/input/frozen_inference_graph.pb",
        "Resources/input/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt",
        "Tensorflow");

    // read the image from disk
    Mat image = img;
    int image_height = image.cols;
    int image_width = image.rows;
    //create blob from image
    Mat blob = blobFromImage(image, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5),
        true, false);
    //create blob from image
    model.setInput(blob);
    //forward pass through the model to carry out the detection
    Mat output = model.forward();

    Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());
    //rect = Rect(10, 30, 100, 100);
    for (int i = 0; i < 1; i++) {
        int class_id = detectionMat.at<float>(i, 1);
        float confidence = detectionMat.at<float>(i, 2);

        // Check if the detection is of good quality
        if (confidence > 0.7 && class_id == 1) {
            int box_x = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
            int box_y = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
            int box_width = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols - box_x);
            int box_height = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows - box_y);
            //rectangle(image, Point(box_x, box_y), Point(box_x + box_width, box_y + box_height), Scalar(255, 255, 255), 2);
            //putText(image, class_names[class_id - 1].c_str(), Point(box_x, box_y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);



            rect = Rect(box_x, box_y, box_width, box_height);



        }
    }
    
   
}


void main(int argc, char** argv) {
    //cv::Mat src = cv::imread("Resources/lambo.png");
    VideoCapture cap("Resources/chaplin.mp4");
    Mat src;


    int maskThreshold = 50;
    cap.read(src);
    cv::Rect boundingBox;
    getBoundingBox(src, boundingBox);

    int frame_num = 0;
    while (true) {


        // Start timer
        double timer = (double)getTickCount();
        cap.read(src);
        frame_num++;
        if (frame_num % 8 == 0) {

            //cv::resize(src, src, Size(640, 480));
            assert(!src.empty());

            

            //    cv::Rect boundingBox(150,80,410,500); //My Image Bounding Box Value

                /* Reading Bounding Box */




            cv::Mat mask = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
            cv::Mat bgModel, fgModel;

            unsigned int iteration = 5; //Tune Parameter according to need
            cv::grabCut(src, mask, boundingBox, bgModel, fgModel, iteration, cv::GC_INIT_WITH_RECT);

            cv::Mat mask2 = (mask == 1) + (mask == 3);  // 0 = cv::GC_BGD, 1 = cv::GC_FGD, 2 = cv::PR_BGD, 3 = cv::GC_PR_FGD
            cv::Mat dest;
            src.copyTo(dest, mask2);

            Mat BlurImg;
            GaussianBlur(src, BlurImg, Size(23, 23), 5, 5);
            Mat maskk = (dest < maskThreshold);

            BlurImg.copyTo(src, maskk);
            // Calculate Frames per second (FPS)
            float fps = getTickFrequency() / ((double)getTickCount() - timer);


            string label = format("FPS : %0.0f ", fps);// Display FPS on frame


            putText(src, label, Point(100, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);


            //imshow("dest", dest);

            imshow("Blured", src);
        }
        
        
        
        cv::waitKey(1);

    }

    destroyAllWindows();

}