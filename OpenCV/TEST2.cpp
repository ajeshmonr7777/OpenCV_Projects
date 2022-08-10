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
#include "IBGS.h"
#include "Multicue.h"






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





int main(int argc, char* argv[])
{

    vector<string> classes;
    vector<Scalar> colors;

    CommandLineParser parser(argc, argv, params);
    parser.about("This program shows how to use background subtraction methods provided by "
        " OpenCV. You can process both videos and images.\n");





  


    if (parser.has("help"))
    {
        //print help information
        parser.printMessage();
    }
    //create Background Subtractor objects
    //bgslibrary::algorithms::MultiCue pBackSub();
    std::shared_ptr<bgslibrary::algorithms::MultiCue> multiCue = std::make_shared<bgslibrary::algorithms::MultiCue>();
  /*  Ptr<BackgroundSubtractor> pBackSub;
    if (parser.get<String>("algo") == "MOG2")
        pBackSub = createBackgroundSubtractorMOG2();
    else*/
        //pBackSub = createBackgroundSubtractorKNN();
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        //error in opening the video input
        cerr << "Unable to open: " << parser.get<String>("input") << endl;
        return 0;
    }
    Mat frame, fgMask, background,brd;
    




    

        

        



    

        


    

    










    


    


    //cvtColor(brd, brd, COLOR_BGR2GRAY);
  
    /*findContours(brd, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    drawContours(brd, contours, -1, Scalar(0, 0, 255), 5);*/

    
    
    
    //imshow("brd", brd);
    Mat element = getStructuringElement(MORPH_ELLIPSE,
        Size(2 * 15 + 1, 2 * 15 + 1),
        Point(5, 5));
    uint frame_number = 0;
    while (true) {
        
        
        cap >> frame;
        resize(frame, frame, Size(640, 480));
        frame_number++;



        if (frame_number % 80 == 1) {


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


            // Open a video file or an image file or a camera stream.
            string str, outputFile;

            VideoWriter video;
            Mat  blob;



            // Start timer
            double timer = (double)getTickCount();

            float frame_count;
            frame_count = cap.get(CAP_PROP_POS_FRAMES);
            cout << "Frame Count = " << frame_count << endl;


            //resize(frame, frame, Size(240, 180));
            // Stop the program if reached end of video
            if (frame.empty()) {
                cout << "Done processing !!!" << endl;
                cout << "Output file is stored as " << outputFile << endl;

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

            resize(brd, brd, Size(640, 480));
            brd = brd < 127;

        }











        if (frame_number < 40 || frame_number % 80 <= 15   ) {
            Mat board(frame.size().height, frame.size().width, CV_8UC3, Scalar(0, 0, 0));
            

            board.copyTo(frame, brd);

           
        }
        
        



        //imshow("Before", frame);






        multiCue->process(frame, fgMask, background);

        Mat BlurImg;
        cv::GaussianBlur(frame, BlurImg, Size(23, 23), 5, 5);

        if (frame_number > 5)
        {
            //cv::imshow("background", background);
            //cv::waitKey(1);





            Mat board2(frame.size().height, frame.size().width, CV_8UC3, Scalar(255, 0, 0));
            //cvtColor(fgMask, fgMask, COLOR_BGR2GRAY);


            threshold(fgMask, fgMask, 30, 255, THRESH_BINARY);
            
            //erode(fgMask, fgMask, element);
            dilate(fgMask, fgMask, element);
            erode(fgMask, fgMask, element);
            dilate(fgMask, fgMask, element);
            erode(fgMask, fgMask, element);
            dilate(fgMask, fgMask, element);
            erode(fgMask, fgMask, element);
            
            //GaussianBlur(fgMask, fgMask, Size(7, 7), 5, 5);
            Mat board3(frame.size().height, frame.size().width, CV_8UC3, Scalar(0, 0, 0));

            int mask_threshold = 100;

            Mat Mask = fgMask < mask_threshold;

            Mat Mask2 = Mask.clone();
            
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;

            //Mask.convertTo(Mask, CV_8UC1);
            cvtColor(Mask2, Mask2, COLOR_BGR2GRAY);

            threshold(Mask2, Mask2, 10, 255, THRESH_BINARY_INV);

            findContours(Mask2, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
            



            for (int i = 0; i < contours.size(); i++) {
                int area = contourArea(contours[i]);
                cout << area << endl;

                if (area > 2000) {
                    drawContours(board3, contours, -1, Scalar(255, 255, 255), -1);
                }

            }
            //threshold(fgMask, fgMask, 120, 255, THRESH_BINARY);
            Mat frame2 = frame.clone();

            
            threshold(board3, board3, 10, 255, THRESH_BINARY_INV);

            BlurImg.copyTo(frame, board3);

            board2.copyTo(frame2, board3);

            //cvtColor(Mask, Mask, COLOR_BGR2GRAY);

            //

            cout << Mask.size() << endl;

            

            threshold(Mask, Mask, 10, 255, THRESH_BINARY_INV);
            

            

            //show the current frame and the fg masks
            imshow("FG Mask", Mask);
            imshow("Frame", frame);
            imshow("Frame 2", frame2);
            //imshow("Board 3", board3);
            
            //get the input from the keyboard
            int keyboard = waitKey(30);
            if (keyboard == 'q' || keyboard == 27)
                break;
        }
    }
    return 0;
}
