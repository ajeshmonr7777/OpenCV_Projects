#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;







void main() {

	VideoCapture cap(0);
	Mat img , black, imgCrop, roi1, roi2 ;

	while (true) {
		cap.read(img);
		resize(img, img, Size(640, 480));
		black = Mat::zeros(img.size(), CV_32FC1);
		CascadeClassifier facecascade;
		facecascade.load("Resources/haarcascade_frontalface_default.xml");

		
		vector <Rect> faces;
		facecascade.detectMultiScale(img, faces, 1.1, 10);

		double alpha = 0.9;
		double beta = 0.1;


		if (faces.empty() == false) {
			for (int i = 0;i < faces.size(); i++) {
				rectangle(img, faces[i].tl(), faces[i].br(), Scalar(213, 132, 56), 3);
				rectangle(black, faces[i].tl(), faces[i].br(), Scalar(213, 132, 56), 3);

				

				roi1 = img(faces[i]);
				roi2 = black(faces[i]);

				
				// addWeighted(roi1, alpha, roi2, beta, 0.0, roi2);
				

				

			}
		}

		

		if (imgCrop.empty() == false) {
			imshow("Image Crop", imgCrop);

		}
		imshow("Black", black);
		//imshow("Black", black);
		imshow("WebCam", img);
		
		waitKey(1);
	}
	destroyAllWindows();




}