
#include "opencv2/opencv.hpp"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<sstream>
#include <boost/concept_check.hpp>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{

	// Read image
	//Mat imFullsize = imread( "G_harfi_arazi.jpg", IMREAD_GRAYSCALE );


	//Size size(640, 480);

	//Mat im;
	//resize(imFullsize, im, size);
	Mat im = imread("G_Harfi_arazi.jpg", IMREAD_GRAYSCALE);
	//imshow("Gray Scale Image", im);

	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;

	// Change thresholds
	params.minThreshold = 10;
	params.maxThreshold = 200;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 75;
	params.maxArea = 200;
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

	Rect theLetter(x+4-w,y+4-h,2*w-8,2*h-8);
	
	Mat croppedImg = im(theLetter);

	Mat binaryROI;
	
	cv::threshold(croppedImg, binaryROI, 200, 255, cv::THRESH_BINARY);
	Mat readable = cv::Scalar::all(255) - binaryROI;
	imwrite("roi.jpg",readable);

	waitKey(0);

}

