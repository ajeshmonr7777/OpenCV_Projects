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
//#include "IBGS.h"
//#include "Multicue.h"
#include <opencv2/tracking.hpp>






using namespace cv;
using namespace std;
using namespace dnn;


const char* params
= "{ help h         |           | Print usage }"
"{ input          | vtest.avi | Path to a video or a sequence of image }"
"{ algo           | MOG2      | Background subtraction method (KNN, MOG2) }";







// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float maskThreshold = 0.3; // Mask threshold
Mat process(Mat& frame, const vector<Mat>& outs)
{



    Mat outDetections = outs[0];
    Mat outMasks = outs[1];

    // Output size of masks is NxCxHxW where
    // N - number of detected boxes
    // C - number of classes (excluding background)
    // HxW - segmentation shape
    const int numDetections = outDetections.size[2];
    const int numClasses = outMasks.size[1];

    outDetections = outDetections.reshape(1, outDetections.total() / 7);




    Mat board(frame.size().height, frame.size().width, CV_8U, Scalar(255, 255, 255));

    for (int i = 0; i < 1; ++i)
    {
        float score = outDetections.at<float>(i, 2);
        if (score > confThreshold)
        {
            // Extract the bounding box
            int classId = static_cast<int>(outDetections.at<float>(i, 1));
            int left = static_cast<int>(frame.cols * outDetections.at<float>(i, 3));
            int top = static_cast<int>(frame.rows * outDetections.at<float>(i, 4));
            int right = static_cast<int>(frame.cols * outDetections.at<float>(i, 5));
            int bottom = static_cast<int>(frame.rows * outDetections.at<float>(i, 6));


            if (classId == 0) {
                left = max(0, min(left, frame.cols - 1));
                top = max(0, min(top, frame.rows - 1));
                right = max(0, min(right, frame.cols - 1));
                bottom = max(0, min(bottom, frame.rows - 1));
                Rect box = Rect(left, top, right - left + 1, bottom - top + 1);

                // Extract the mask for the object
                Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i, classId));

                Mat Maskk;
                resize(objectMask, Maskk, Size(box.width, box.height));
                Mat maskk = (Maskk < maskThreshold);


                //Scalar color = colors[classId + 1 % colors.size()];

                //Resize the mask, threshold, color and apply it on the image
                resize(objectMask, objectMask, Size(box.width, box.height));
                Mat mask = (objectMask < maskThreshold);

                float alpha = 1.0;
                float beta = 0.0;



                Mat roi1 = mask;
                Mat roi2 = board(box);


                addWeighted(roi1, alpha, roi2, beta, 0.0, roi2);



                //imshow("Board", board);



            }

        }
    }
    return board;


}







Mat MaskRCNN(Mat frame, Mat blob) {

    



    vector<string> classes;
    vector<Scalar> colors;
    Mat brd;


    // Load names of classes
    string classesFile = "mscoco_labels.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);



    // Load the colors
    string colorsFile = "colors.txt";
    ifstream colorFptr(colorsFile.c_str());
    while (getline(colorFptr, line)) {
        char* pEnd;
        double r, g, b;
        r = strtod(line.c_str(), &pEnd);
        g = strtod(pEnd, NULL);
        b = strtod(pEnd, NULL);
        Scalar color = Scalar(r, g, b, 255.0);
        colors.push_back(Scalar(r, g, b, 255.0));
    }

    // Give the configuration and weight files for the model
    String textGraph = "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
    String modelWeights = "frozen_inference_graph.pb";

    // Load the network
    Net net = readNetFromTensorflow(modelWeights, textGraph);


    cout << "Using CPU device" << endl;
    net.setPreferableBackend(DNN_TARGET_CPU);


    /*cout << "Using GPU device" << endl;
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);*/









    //resize(frame, frame, Size(240, 180));
    // Stop the program if reached end of video
    if (frame.empty()) {
        cout << "Done processing !!!" << endl;
        //cout << "Output file is stored as " << outputFile << endl;

    }
    // Create a 4D blob from a frame.
    blobFromImage(frame, blob, 1.0, Size(frame.cols, frame.rows), Scalar(), true, false);
    //blobFromImage(frame, blob);

    //Sets the input to the network
    net.setInput(blob);

    // Runs the forward pass to get output from the output layers
    std::vector<String> outNames(2);
    outNames[0] = "detection_out_final";
    outNames[1] = "detection_masks";
    vector<Mat> outs;
    net.forward(outs, outNames);

    // Extract the bounding box and mask for each of the detected objects
    brd = process(frame, outs);



    
    return brd;
}




int main(int argc, char* argv[])
{

    

    CommandLineParser parser(argc, argv, params);
    parser.about("This program shows how to use background subtraction methods provided by "
        " OpenCV. You can process both videos and images.\n");








    if (parser.has("help"))
    {
        //print help information
        parser.printMessage();
    }
;
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        //error in opening the video input
        cerr << "Unable to open: " << parser.get<String>("input") << endl;
        return 0;
    }
    Mat frame, fgMask, background, brd;




































    //cvtColor(brd, brd, COLOR_BGR2GRAY);

    /*findContours(brd, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    drawContours(brd, contours, -1, Scalar(0, 0, 255), 5);*/




    //imshow("brd", brd);
    Mat element = getStructuringElement(MORPH_ELLIPSE,
        Size(2 * 15 + 1, 2 * 15 + 1),
        Point(5, 5));
    uint frame_number = 0;




    Ptr<Tracker> tracker;

    Rect2d bbox = Rect2d (5,5,20,20);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    Mat window;

    
    bool ok = false;
    while (true) {


        cap >> frame;
        resize(frame, frame, Size(640, 480));

        Mat frame_new = frame.clone();


        Mat BlurImg, BlurImg2;
        cv::GaussianBlur(frame, BlurImg, Size(23, 23), 5, 5);
        BlurImg2 = BlurImg.clone();

        Mat black_board(frame.size().height, frame.size().width, CV_8UC3, Scalar(0, 0, 0));
        Mat black_board2 = black_board.clone();

        Mat board(frame.size().height, frame.size().width, CV_8UC3, Scalar(0, 0, 0));


        

        
        

        if (frame_number==0  ) {


            


            // Open a video file or an image file or a camera stream.
            string str, outputFile;

            VideoWriter video;
            Mat  blob;



            brd = MaskRCNN(frame, blob);

         




            resize(brd, brd, Size(640, 480));
            brd = brd < 127;// get bboxes from the contours and store them in vetor of rectangle

            

            //Mask.convertTo(Mask, CV_8UC1);
            //cvtColor(brd, brd, COLOR_BGR2GRAY);

            //imshow("brd", brd);

            findContours(brd, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

            bbox = Rect(10, 10, 20, 20);

            if (contours.size() != 0) {
                bbox = boundingRect(contours[0]);
            }
            


            window = brd(bbox);

            cvtColor(window, window, COLOR_GRAY2BGR);

            tracker = TrackerMIL::create();
            tracker->init(frame, bbox);
            //cout << bbox << endl;


            rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);



            

        }
        else
        {
            // use tracker when you are not using rcnn
            // pass the above vector of rectangle to the tracker and update their postiions in new vector of rectangle
            // Update the tracking result
            rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
            ok = tracker->update(frame, bbox);
            
            if (ok)
            {
                // Tracking success : Draw the tracked object
                rectangle(frame, bbox, Scalar(0, 255, 0), 2, 1);


                //resize(window, window, Size(bbox.width, bbox.height));

                //
                //

                float alpha = 1.0;
                float beta = 0.0;



                //Mat roi1 = window;
                //Mat roi2 = black_board(bbox);


                //addWeighted(roi1, alpha, roi2, beta, 0.0, roi2);


                
                
                Mat white_box(bbox.height, bbox.width, CV_8UC3, Scalar(255, 255, 255));

                Mat roi3 = white_box;
                Mat roi4 = black_board2(bbox);

                addWeighted(roi3, alpha, roi4, beta, 0.0, roi4);
                
            }
            else
            {
                // Tracking failure detected.
                putText(frame, "Tracking failure detected", Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
                putText(board, "Tracking failure detected", Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);

               
            }

            
        }








        


        if (frame_number > 1) {
            


            frame_new.copyTo(board, black_board2);
            frame_new.copyTo(BlurImg, black_board);
            frame_new.copyTo(BlurImg2, black_board2);
            
            //frame.copyTo(board, brd);
            //frame.copyTo(BlurImg, brd);
            //frame.copyTo(board, black_board);


        }

        



        //imshow("Before", frame);











            //show the current frame and the fg masks
            //imshow("FG Mask", Mask);
          // imshow("Frame", frame);
            //imshow("Frame 2", frame2);
          // imshow("Board", board);
           imshow("BBlur", BlurImg2);
           //imshow("BBlur2", BlurImg2);

           // imshow("name", frame(bbox));

            //get the input from the keyboard
            int keyboard = waitKey(30);
            if (keyboard == 'q' || keyboard == 27)
                break;
     


            frame_number++;
    }
    return 0;
}
