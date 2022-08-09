#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
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
using namespace dnn;
using namespace std;

// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()



Rect getBoundingBox(cv::Mat& img, cv::Rect2d& rect)
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
    return rect;
}








int main(int argc, char** argv)
{
    // List of tracker types in OpenCV 3.2
    // NOTE : GOTURN implementation is buggy and does not work.
    string trackerTypes[7] = { "BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "CSRT" };
    // vector <string> trackerTypes(types, std::end(types));

    // Create a tracker
    string trackerType = trackerTypes[2];

    Ptr<Tracker> tracker;

#if (CV_MINOR_VERSION < 3)
    {
        tracker = Tracker::create(trackerType);
    }
#else
    {
        if (trackerType == "BOOSTING")
            tracker = TrackerBoosting::create();
        if (trackerType == "MIL")
            tracker = TrackerMIL::create();
        if (trackerType == "KCF")
            tracker = TrackerKCF::create();
        if (trackerType == "TLD")
            tracker = TrackerTLD::create();
        if (trackerType == "MEDIANFLOW")
            tracker = TrackerMedianFlow::create();
        if (trackerType == "GOTURN")
            tracker = TrackerGOTURN::create();
        if (trackerType == "CSRT")
            tracker = TrackerCSRT::create();
    }
#endif
    // Read video
    VideoCapture video("Resources/chaplin.mp4");

    // Exit if video is not opened
    if (!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        return 1;

    }

    // Read first frame
    Mat frame;
    bool ok = video.read(frame);

    // Define initial boundibg box
    /*Rect2d bbox(287, 23, 86, 320);*/

    // Uncomment the line below to select a different bounding box
    //Rect2d bbox = selectROI(frame, false);
    Rect2d bbox;
    getBoundingBox(frame, bbox);


    // Display bounding box.
    rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
    imshow("Tracking", frame);

    tracker->init(frame, bbox);

    while (video.read(frame))
    {

        // Start timer
        double timer = (double)getTickCount();

        // Update the tracking result
        bool ok = tracker->update(frame, bbox);

        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);


        if (ok)
        {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
        }
        else
        {
            // Tracking failure detected.
            putText(frame, "Tracking failure detected", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        }

        // Display tracker type on frame
        putText(frame, trackerType + " Tracker", Point(100, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

        string label = format("FPS : %0.0f ", fps);// Display FPS on frame


        putText(frame, label , Point(100, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

        // Display frame.
        imshow("Tracking", frame);

        // Exit if ESC pressed.
        int k = waitKey(1);
        if (k == 27)
        {
            break;
        }

    }



}