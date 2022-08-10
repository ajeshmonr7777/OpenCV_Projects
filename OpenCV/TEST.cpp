#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

using namespace cv;
using namespace std;
using namespace dnn;


const char* params
= "{ help h         |           | Print usage }"
"{ input          | vtest.avi | Path to a video or a sequence of image }"
"{ algo           | MOG2      | Background subtraction method (KNN, MOG2) }";
int main(int argc, char* argv[])
{
    CommandLineParser parser(argc, argv, params);
    parser.about("This program shows how to use background subtraction methods provided by "
        " OpenCV. You can process both videos and images.\n");





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



    if (parser.has("help"))
    {
        //print help information
        parser.printMessage();
    }
    //create Background Subtractor objects
    Ptr<BackgroundSubtractor> pBackSub;
    if (parser.get<String>("algo") == "MOG2")
        pBackSub = createBackgroundSubtractorMOG2();
    else
        pBackSub = createBackgroundSubtractorKNN();
    VideoCapture capture(0);
    if (!capture.isOpened()) {
        //error in opening the video input
        cerr << "Unable to open: " << parser.get<String>("input") << endl;
        return 0;
    }
    Mat frame, fgMask,background;
    uint frame_number = 0;

    

    while (true) {
        cap >> frame;
        Mat frame0 = frame.clone();
        if (frame.empty())
            break;
        //update the background model
        frame_number++;

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

        Mat roi = frame;
        Rect box;
        
        Mat board(frame.size().height, frame.size().width, CV_8UC3, Scalar(0, 0, 0));
        box = Rect(10, 20, 200, 200);
        for (int i = 0; i < 1; i++) {
            int class_id = detectionMat.at<float>(i, 1);
            float confidence = detectionMat.at<float>(i, 2);

            // Check if the detection is of good quality
            if (confidence > 0.2 && class_id ==1) {
                int box_x = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int box_y = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int box_width = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols - box_x);
                int box_height = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows - box_y);
                rectangle(frame, Point(box_x, box_y), Point(box_x + box_width, box_y + box_height), Scalar(255, 0, 255), 2);
                //putText(frame, class_names[class_id - 1].c_str(), Point(box_x, box_y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);

                //box = Rect(10, 20, 200, 200);
                box = Rect(box_x, box_y, box_width, box_height);
                

                float alpha = 1.0;
                float beta = 0.0;

                Mat roi1 = board(box);
                Mat roi2 = frame0(box);

                addWeighted(roi1, alpha, roi2, beta, 0.0, roi2);
                
            }
        }

        
        




        imshow("name", frame0);




     

        pBackSub->apply(frame, fgMask);
        
        if (frame_number > 100)
        {
            pBackSub->getBackgroundImage(background);
            cv::imshow("background", background);
            cv::waitKey(1);


            Mat element = getStructuringElement(MORPH_ELLIPSE,
                Size(2 * 3 + 1, 2 * 3 + 1),
                Point(5, 5));


            Mat board2(frame.size().height, frame.size().width, CV_8UC3, Scalar(255, 0, 0));
            //cvtColor(fgMask, fgMask, COLOR_BGR2GRAY);
            dilate(fgMask, fgMask, element);
            erode(fgMask, fgMask, element);
            //dilate(fgMask, fgMask, element);
            //erode(fgMask, fgMask, element);

            int mask_threshold = 100;

            Mat Mask = fgMask < mask_threshold;

            //threshold(fgMask, fgMask, 120, 255, THRESH_BINARY);
            Mat frame2 = frame.clone();

            Mat BlurImg;
            cv::GaussianBlur(frame, BlurImg, Size(23, 23), 5, 5);

            BlurImg.copyTo(frame, Mask);

            board2.copyTo(frame2, Mask);




            //show the current frame and the fg masks
            imshow("Frame", frame);
            imshow("Frame 2", frame2);
            imshow("FG Mask", Mask);
            //get the input from the keyboard
            int keyboard = waitKey(30);
            if (keyboard == 'q' || keyboard == 27)
                break;
        }
    }
    return 0;
}