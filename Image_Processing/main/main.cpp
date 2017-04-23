#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <stdio.h>
#include </usr/include/tesseract/baseapi.h>
#include </usr/include/leptonica/allheaders.h>

//#include<Windows.h>

using namespace cv;
using namespace std;

//functions prototypes
void on_trackbar(int, void*);
void createTrackbars();
void showimgcontours(Mat &threshedimg, Mat &original);
void toggle(int key);
void morphit(Mat &img);
void blurthresh(Mat &img);

//function prototypes ends here

//boolean toggles

bool domorph = false;
bool doblurthresh = false;
bool showchangedframe = false;
bool showcontours = false;

//boolean toggles end

frameX = 1920;
frameY = 1080;


int H_MIN = 158; // 0
int H_MAX = 255;
int S_MIN = 20; // 0
int S_MAX = 255;
int V_MIN = 118; // 0
int V_MAX = 255;

int kerode = 1;
int kdilate = 1;
int kblur = 1;
int threshval = 0;

int i;
int angle;
char *outText;
char textArray[35];
PIX *src;

int main(void)
{
	// Init tesseract
	tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
    // Initialize tesseract-ocr with English, without specifying tessdata path
    if (api->Init(NULL, "eng")) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
    }

	Mat frame, hsvframe, rangeframe;
	Mat im;
	int key;
	VideoCapture cap("ImageProcessFlight_01.mp4");
	while ((key = waitKey(30)) != 27)
	{
		cap >> frame;
		//cv::resize(rawFrame, frame, cv::Size(640, 480));
		//flip(frame, frame, 180);
		//frame = imread("G_Harfi_arazi.jpg");

		cvtColor(frame, hsvframe, COLOR_BGR2HSV);


		inRange(hsvframe, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), im);

		
			SimpleBlobDetector::Params params;

			// Change thresholds
			params.minThreshold = 10;
			params.maxThreshold = 200;

			// Filter by Area.
			params.filterByArea = true;
			params.minArea = 500;
			//params.maxArea = 200;
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

			Mat im_with_keypoints;
			drawKeypoints( im, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

			// Show blobs
			imshow("Blob Countours", im_with_keypoints );
			//int x, y,w,h;
			int x = 0, y = 0,w = 0,h = 0;

			for (std::vector<cv::KeyPoint>::iterator blobIterator = keypoints.begin(); blobIterator != keypoints.end(); blobIterator++) {
				std::cout << "Size of Letter is: " << blobIterator->size << std::endl;
				std::cout << "Letter is at: " << blobIterator->pt.x << " " << blobIterator->pt.y << std::endl;
				x = blobIterator->pt.x;
				y = blobIterator->pt.y;
				w = blobIterator->size;
				h = blobIterator->size;
			}


			if (x+1.5*w < frameX && y+1.5*h < frameY && x-1.5*w > 0 && y-1.5*h > 0) // if there is enough pixels
			{
				Rect theLetter(x-1.5*w,y-1.5*h,3*w,3*h);
				Mat croppedImg = im(theLetter);
				imwrite("roi.jpg",croppedImg);
				cv::Mat cv_src = cv::imread("roi.jpg");
				angle = 0;
				for(i=1; i <= 36; i++)
				    {
				        angle = i*10;
				        cv::Point2f center(cv_src.cols/2.0, cv_src.rows/2.0);
				        cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
				        // determine bounding rectangle
				        cv::Rect bbox = cv::RotatedRect(center,cv_src.size(), angle).boundingRect();
				        // adjust transformation matrix
				        rot.at<double>(0,2) += bbox.width/2.0 - center.x;
				        rot.at<double>(1,2) += bbox.height/2.0 - center.y;

				        cv::Mat dst;
				        cv::warpAffine(cv_src, dst, rot, bbox.size());

				        PIX *src = pixCreate(dst.size().width, dst.size().height, 8);

				        for(int i=0; i<dst.rows; i++) {
				            for(int j=0; j<dst.cols; j++) {
				                pixSetPixel(src, j,i, (l_uint32) dst.at<uchar>(i,j));
				            }
				        }

				        api->SetImage(src);
				        // Get OCR result
				        outText = api->GetUTF8Text();
				        textArray[i-1] = *outText;
				        //printf("OCR output:\n%s", outText);
				        // get rotation matrix for rotating the image around its center
				    }

				    			    // Find the mos occured character
			    int max=0;
			    char maxCharacter;
			    int count;
			    for(char q='A';q<='Z';q++)
			    {
			        count=0;
			        for(i=0; i<strlen(textArray);i++)
			        {
			            if(textArray[i]==q)
			                count++;
			        }

			        if(count>max)
			        {
			            max=count;
			            maxCharacter=q;
			        }
			    }
			    std::cout << "Letter is: " << maxCharacter << std::endl;
			    pixDestroy(&src);
			    for (i=0; i <36; i++)
			    {
			    	textArray[i] = 0;
			    }

			}



			
	}
	api->End();
	return 0;
}