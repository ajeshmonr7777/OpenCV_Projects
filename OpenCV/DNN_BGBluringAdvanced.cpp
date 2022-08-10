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




const char* keys =
"{help h usage ? | | Usage examples: \n\t\t./mask-rcnn.out --image=traffic.jpg \n\t\t./mask-rcnn.out --video=sample.mp4}"
"{image i        |<none>| input image   }"
"{video v       |<none>| input video   }"
"{device d       |<none>| device }"
;
using namespace cv;
using namespace dnn;
using namespace std;

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float maskThreshold = 0.3; // Mask threshold

vector<string> classes;
vector<Scalar> colors;

// Draw the predicted bounding box
//void drawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask);

// Postprocess the neural network's output for each frame
//void postprocess(Mat& frame, const vector<Mat>& outs);

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    // Load names of classes
    string classesFile = "mscoco_labels.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    string device = parser.get<String>("device");

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

   
    /*cout << "Using CPU device" << endl;
    net.setPreferableBackend(DNN_TARGET_CPU);*/
   
   
    cout << "Using GPU device" << endl;
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);
    

    // Open a video file or an image file or a camera stream.
    string str, outputFile;
    VideoCapture cap(0);
    VideoWriter video;
    Mat frame, blob;







    // Create a window
    //static const string kWinName = "Deep learning object detection in OpenCV";
   // namedWindow(kWinName, WINDOW_NORMAL);

    // Process frames.
    while (waitKey(1) < 0)
    {
        // get frame from the video
        cap >> frame;

        float frame_count;
        frame_count = cap.get(CAP_PROP_POS_FRAMES);
        cout << "Frame Count = " << frame_count << endl;
        

        //resize(frame, frame, Size(240, 180));
        // Stop the program if reached end of video
        if (frame.empty()) {
            cout << "Done processing !!!" << endl;
            cout << "Output file is stored as " << outputFile << endl;
            waitKey(3000);
            break;
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
        //postprocess(frame, outs);




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

        for (int i = 0; i < numDetections; ++i)
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

                    

                    //imshow("Mask", maskk);


                    

                    



                    Scalar color = colors[classId + 1 % colors.size()];

                    //Resize the mask, threshold, color and apply it on the image
                    resize(objectMask, objectMask, Size(box.width, box.height));
                    Mat mask = (objectMask < maskThreshold);

                    float alpha = 1.0;
                    float beta = 0.0;

                    

                    Mat roi1 = mask;
                    Mat roi2 = board(box);


                    addWeighted(roi1, alpha, roi2, beta, 0.0, roi2);

                    imshow("Board", board);

                    /*Mat coloredRoi = (0.9 * color + 0.1 * frame);*/
                    Mat coloredRoi;
                    GaussianBlur(frame, coloredRoi, Size(19,19), 5, 5);
                    coloredRoi.convertTo(coloredRoi, CV_8UC3);

                    // Draw the contours on the image
                    vector<Mat> contours;
                    Mat hierarchy;
                    board.convertTo(board, CV_8U);
                    findContours(board, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
                    //drawContours(coloredRoi, contours, -1, color, 1, LINE_8, hierarchy, 100);
                    coloredRoi.copyTo(frame, board);



                    Mat newFrame = frame(box);

                    

                    
                }

            }
        }

        



        // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("Inference time for a frame : %0.0f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0));


        resize(frame, frame, Size(1000, 700));

        imshow("NewFrame", frame);

        // Write the frame with the detection boxes
        Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        if (parser.has("image")) imwrite(outputFile, detectedFrame);
        else video.write(detectedFrame);

        //imshow(kWinName, frame);

        

    }

    cap.release();
    if (!parser.has("image")) video.release();

    return 0;
}



// Draw the predicted bounding box, colorize and show the mask on the image
