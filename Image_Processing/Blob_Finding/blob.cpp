
#include "opencv2/opencv.hpp"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<sstream>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{

	// Read image
	Mat imFullsize = imread( "yeniharfler.png", IMREAD_GRAYSCALE );


	Size size(640, 480);

	Mat im;
	resize(imFullsize, im, size);
	//Mat im = imread("blob.jpg", IMREAD_GRAYSCALE);

	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;

	// Change thresholds
	params.minThreshold = 10;
	params.maxThreshold = 200;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 175;
	params.maxArea = 450;
	//params.filterByArea = false;

	// Filter by Circularity
	//params.filterByCircularity = true;
	//params.minCircularity = 0.1;
	params.filterByCircularity = false;

	// Filter by Convexity
	//params.filterByConvexity = true;
	//params.minConvexity = 0.87;
	params.filterByConvexity = false;


	// Filter by Inertia
	//params.filterByInertia = true;
	//params.minInertiaRatio = 0.01;
	params.filterByInertia = false;

	// Storage for blobs
	vector<KeyPoint> keypoints;


#if CV_MAJOR_VERSION < 3   // If you are using OpenCV 2

	// Set up detector with params
	SimpleBlobDetector detector(params);

	// Detect blobs
	detector.detect( im, keypoints);
#else 

	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);   

	// Detect blobs
	detector->detect( im, keypoints);
#endif 

	// Draw detected blobs as red circles.
	// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
	// the size of the circle corresponds to the size of blob

	Mat im_with_keypoints;
	drawKeypoints( im, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

	// Show blobs
	imshow("Blob Countours", im_with_keypoints );
	int x, y,w,h;
	for (std::vector<cv::KeyPoint>::iterator blobIterator = keypoints.begin(); blobIterator != keypoints.end(); blobIterator++) {
		std::cout << "size of blob is: " << blobIterator->size << std::endl;
		std::cout << "point is at: " << blobIterator->pt.x << " " << blobIterator->pt.y << std::endl;
		x = blobIterator->pt.x;
		y = blobIterator->pt.y;
		w = blobIterator->size;
		h = blobIterator->size;
	}


	SimpleBlobDetector::Params params2;

	// Change thresholds
	//params2.minThreshold = 1;
	//params2.maxThreshold = 255;

	// Filter by Area.
	//params2.filterByArea = true;
	//params2.minArea = 150;
	//params.maxArea = 450;
	//params.filterByArea = false;

	// Filter by Circularity
	//params.filterByCircularity = true;
	//params.minCircularity = 0.1;
	params2.filterByCircularity = false;

	// Filter by Convexity
	//params2.filterByConvexity = true;
	//params2.minConvexity = 0.87;
	params2.filterByConvexity = false;


	// Filter by Inertia
	//params.filterByInertia = true;
	//params.minInertiaRatio = 0.01;
	params2.filterByInertia = false;

	// Storage for blobs
	vector<KeyPoint> keypoints2;

	Ptr<SimpleBlobDetector> detector2 = SimpleBlobDetector::create(params2);

	// Detect blobs
	detector2 -> detect(im, keypoints2);

	Mat im2_with_keypoints;
	drawKeypoints(im, keypoints, im2_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imshow("Blob Countours 2", im2_with_keypoints);


	/*
	Rect roi = Rect(x-w/2, y-h/2, w, h);
	Mat image_roi = im(roi);

	Mat img_to_read = image_roi < 200;

	imshow("Roi", image_roi);
	imshow("The Letter", img_to_read);

	imwrite("Detection.jpg", img_to_read);
	
	Mat cvMat = image_roi;
	//cv::cvtColor(cvMat, cvMat, CV_RGB2GRAY);
	// Apply adaptive threshold.
	cv::adaptiveThreshold(cvMat, cvMat, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 3, 5);
	// Attempt to sharpen the image.
	//cv::GaussianBlur(cvMat, cvMat, cv::Size(0, 0), 3);
	cv::addWeighted(cvMat, 1.5, cvMat, -0.5, 0, cvMat);

	imshow("New", image_roi);
	///////////////////////////////////////// BLOB END *** ///////////////////////////////////////////////////////////
	*/

	waitKey(0);

}

