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

#include "IBGS.h"
#include "Multicue.h"





using namespace cv;
using namespace dnn;
using namespace std;

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float maskThreshold = 0.3; // Mask threshold

vector<string> classes;
vector<Scalar> colors;






// Process Mask RCNN frames
void process(Mat& frame, const vector<Mat>& outs);




void showImg(const std::string& name, cv::Mat& img) {
    cv::imshow(name, img);
    cv::waitKey(0);
}



Rect track_Box(Rect rect) {
    // List of tracker types in OpenCV 3.2
    // NOTE : GOTURN implementation is buggy and does not work.
    string trackerTypes[7] = { "BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "CSRT" };
    // vector <string> trackerTypes(types, std::end(types));

    // Create a tracker
    string trackerType = trackerTypes[1];

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
    VideoCapture video(0);

    // Exit if video is not opened
    if (!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        

    }

    // Read first frame
    Mat frame;
    bool ok = video.read(frame);

    // Define initial boundibg box
    /*Rect2d bbox(287, 23, 86, 320);*/

    // Uncomment the line below to select a different bounding box
   
    Rect2d bbox = rect;

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

        

        // Exit if ESC pressed.
        int k = waitKey(1);
        if (k == 27)
        {
            break;
        }

    }
    return bbox;
}




Rect getBoundingBox(cv::Mat& img, cv::Rect& rect)
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
        if (confidence > 0.7 && class_id == 1 ) {
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



void MaskRCNN() {

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


    int frame_num = 0;
    while (waitKey(1) < 0)
    {
        // get frame from the video
        cap >> frame;
        frame_num++;

        if (frame_num % 1 == 0) {



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
            process(frame, outs);



            // Calculate Frames per second (FPS)
            float fps = getTickFrequency() / ((double)getTickCount() - timer);


            string label = format("FPS : %0.0f ", fps); // Display FPS on frame

            resize(frame, frame, Size(1000, 700));

            putText(frame, label, Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));



            imshow("NewFrame", frame);

        }


    }

    cap.release();

}







void grabCut() {

 





    string trackerTypes[7] = { "BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "CSRT" };
    // vector <string> trackerTypes(types, std::end(types));

    // Create a tracker
    string trackerType = trackerTypes[4];

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
    VideoCapture cap(0);


    // Read first frame
    
    Mat src;


    int maskThreshold = 50;
    bool ok = cap.read(src);
    cv::Rect boundingBox;
    getBoundingBox(src, boundingBox);




    Rect2d bbox = boundingBox;

    //tracker->init(src, bbox);




    int frame_num = 0;
    while (true) {


        // Start timer
        

        double timer = (double)getTickCount();
        cap.read(src);

        cuda::GpuMat srcGpu;

        srcGpu.upload(src);


        if (!srcGpu.empty()) {
            cout << "Could not load in image on GPU!" << endl;
        }

        frame_num++;
        if (frame_num % 1 == 0) {

            //cv::resize(src, src, Size(640, 480));
            assert(!src.empty());


            //// Update the tracking result
            //bool ok = tracker->update(src, bbox);
            

            //boundingBox = bbox;





            cv::Mat mask = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
            cv::Mat bgModel, fgModel;

            unsigned int iteration = 5; //Tune Parameter according to need
            cv::grabCut(src, mask, boundingBox, bgModel, fgModel, iteration, cv::GC_INIT_WITH_RECT);

            cv::Mat mask2 = (mask == 1) + (mask == 3);  // 0 = cv::GC_BGD, 1 = cv::GC_FGD, 2 = cv::PR_BGD, 3 = cv::GC_PR_FGD
            cv::Mat dest;
            src.copyTo(dest, mask2);

            Mat BlurImg;
            cv::GaussianBlur(src, BlurImg, Size(23, 23), 5, 5);
            Mat maskk = (dest < maskThreshold);

            BlurImg.copyTo(src, maskk);
            // Calculate Frames per second (FPS)
            float fps = getTickFrequency() / ((double)getTickCount() - timer);


            string label = format("FPS : %0.0f ", fps);// Display FPS on frame


            putText(src, label, Point(100, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);


           // imshow("dest", dest);

            imshow("Blured", src);
        }



        cv::waitKey(1);

    }


}







const char* params
= "{ help h         |           | Print usage }"
"{ input          | vtest.avi | Path to a video or a sequence of image }"
"{ algo           | MOG2      | Background subtraction method (KNN, MOG2) }";

int multicue() {






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



   
    //create Background Subtractor objects
    //bgslibrary::algorithms::MultiCue pBackSub();
    std::shared_ptr<bgslibrary::algorithms::MultiCue> multiCue = std::make_shared<bgslibrary::algorithms::MultiCue>();
    /*  Ptr<BackgroundSubtractor> pBackSub;
      if (parser.get<String>("algo") == "MOG2")
          pBackSub = createBackgroundSubtractorMOG2();
      else*/
      //pBackSub = createBackgroundSubtractorKNN();
    VideoCapture capture(0);
    if (!capture.isOpened()) {
        //error in opening the video input
        cerr << "Unable to open: "  << endl;
        return 0;
    }
    Mat frame, fgMask, background;







    cap >> frame;

    resize(frame, frame, Size(640, 480));


    //update the background model


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


    Rect box(0, 0, 0, 0);


    for (int i = 0; i < 1; i++) {
        int class_id = detectionMat.at<float>(i, 1);
        float confidence = detectionMat.at<float>(i, 2);

        // Check if the detection is of good quality
        if (confidence > 0.3 && class_id == 1) {
            int box_x = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            int box_y = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            int box_width = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols - box_x);
            int box_height = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows - box_y);
            //rectangle(frame, Point(box_x, box_y), Point(box_x + box_width, box_y + box_height), Scalar(255, 0, 255), 2);
            //putText(frame, class_names[class_id - 1].c_str(), Point(box_x, box_y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);


            box = Rect(box_x - 10, box_y - 10, box_width - 10, box_height - 10);




        }
    }




    Mat element = getStructuringElement(MORPH_ELLIPSE,
        Size(2 * 3 + 1, 2 * 3 + 1),
        Point(5, 5));
    uint frame_number = 0;
    while (true) {


        cap >> frame;
        resize(frame, frame, Size(640, 480));
        frame_number++;
        if (frame_number <= 200 && box.width > 0 && box.height > 0) {
            Mat board(frame.size().height, frame.size().width, CV_8UC3, Scalar(0, 0, 0));
            Mat frame0 = frame.clone();

            float alpha = 1.0;
            float beta = 0.0;

            Mat roi1 = board(box);
            Mat roi2 = frame(box);

            addWeighted(roi1, alpha, roi2, beta, 0.0, roi2);
        }





        imshow("Before", frame);






        multiCue->process(frame, fgMask, background);

        Mat BlurImg;
        cv::GaussianBlur(frame, BlurImg, Size(23, 23), 5, 5);

        if (frame_number > 100)
        {
            //cv::imshow("background", background);
            //cv::waitKey(1);





            Mat board2(frame.size().height, frame.size().width, CV_8UC3, Scalar(255, 0, 0));
            //cvtColor(fgMask, fgMask, COLOR_BGR2GRAY);

            erode(fgMask, fgMask, element);
            dilate(fgMask, fgMask, element);
            erode(fgMask, fgMask, element);
            //erode(fgMask, fgMask, element);
            //dilate(fgMask, fgMask, element);

            //GaussianBlur(fgMask, fgMask, Size(7, 7), 5, 5);


            int mask_threshold = 100;

            Mat Mask = fgMask < mask_threshold;

            //threshold(fgMask, fgMask, 120, 255, THRESH_BINARY);
            Mat frame2 = frame.clone();




            BlurImg.copyTo(frame, Mask);

            board2.copyTo(frame2, Mask);

            //cvtColor(Mask, Mask, COLOR_BGR2GRAY);

            //

            cout << Mask.size() << endl;

            threshold(Mask, Mask, 127, 255, THRESH_BINARY_INV);


            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;


            //findContours(Mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
            //drawContours(Mask, contours, 1, Scalar(255, 0, 255), 2);

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








int main(int argc, char** argv)
{
   
    //MaskRCNN();
   

    //grabCut();

    multicue();

    return 0;
}






















void process(Mat& frame, const vector<Mat>& outs)
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
                GaussianBlur(frame, coloredRoi, Size(19, 19), 5, 5);
                coloredRoi.convertTo(coloredRoi, CV_8UC3);

                // Draw the contours on the image
                vector<Mat> contours;
                Mat hierarchy;
                board.convertTo(board, CV_8U);
                findContours(board, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
                //drawContours(coloredRoi, contours, -1, color, 1, LINE_8, hierarchy, 100);
                coloredRoi.copyTo(frame, board);


            }

        }
    }



}