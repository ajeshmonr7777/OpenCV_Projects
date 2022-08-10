#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

using namespace std;
using namespace cv;
using namespace dnn;







// Fill the vector with random colors
void getRandomColors(vector<Scalar>& colors, int numColors)
{
    RNG rng(0);
    for (int i = 0; i < numColors; i++)
        colors.push_back(Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
}



int main(int, char**) {
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
        "TensorFlow");

    // capture the video
    VideoCapture cap(0);
    // get the video frames' width and height for proper saving of videos
    int frame_width = static_cast<int>(cap.get(3));
    int frame_height = static_cast<int>(cap.get(4));
    // create the `VideoWriter()` object
    /*VideoWriter out("Resources/outputs/video_result.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30,
        Size(frame_width, frame_height))*/;


    int frame_number = 0;
    vector<Rect> bbox;
    int box_x, box_y, box_width, box_height;


    Ptr<Tracker> tracker;

    tracker = TrackerBoosting::create();


    // Create multitracker
    Ptr<MultiTracker> multiTracker = cv::MultiTracker::create();







    Mat frame;
    bool isSuccess = cap.read(frame);

    //// Start timer
    //double timer = (double)getTickCount();

        

    int image_height = frame.cols;
    int image_width = frame.rows;



        


    //create blob from image
    Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5),
        true, false);
    //create blob from image
    model.setInput(blob);
    //forward pass through the model to carry out the detection
    Mat output = model.forward();

    Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++) {
        int class_id = detectionMat.at<float>(i, 1);
        float confidence = detectionMat.at<float>(i, 2);

        // Check if the detection is of good quality
        if (confidence > 0.4 && class_id == 1) {
            box_x = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            box_y = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            box_width = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols - box_x);
            box_height = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows - box_y);
            rectangle(frame, Point(box_x, box_y), Point(box_x + box_width, box_y + box_height), Scalar(255, 0, 255), 2);
            //putText(image, class_names[class_id - 1].c_str(), Point(box_x, box_y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
            bbox[i] == Rect(box_x, box_y, box_width, box_height);
        }



    }



        for (int i = 0; i < bbox.size(); i++)
            multiTracker->add(tracker, frame, Rect2d(bbox[i]));

        






        while (cap.isOpened()) {
            

            cap >> frame;
            
            multiTracker->update(frame);

            vector<Scalar> colors;
            getRandomColors(colors, bbox.size());

            // draw tracked objects
            for (unsigned i = 0; i < multiTracker->getObjects().size(); i++)
            {
                rectangle(frame, multiTracker->getObjects()[i], colors[i], 2, 1);
            }

                
            
            // Calculate Frames per second (FPS)
            //float fps = getTickFrequency() / ((double)getTickCount() - timer);


            //string label = format("FPS : %0.0f ", fps); // Display FPS on frame


            //putText(frame, label, Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));


            imshow("image", frame);

            
            //out.write(image);
            int k = waitKey(1);
            if (k == 113) {
                break;
            }

            
        }

        cap.release();
        
}