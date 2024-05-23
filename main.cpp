#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
using namespace std;
using namespace cv;

void detectAndDisplay(Mat frame);
CascadeClassifier face_cascade;
String face_cascade_name;

int main(int argc, const char** argv)
{
    CommandLineParser parser(argc, argv, "{camera|0|Camera device number.}");

    // Load the cascades
    face_cascade_name = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml";
    if(!face_cascade.load(face_cascade_name))
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };

    // Read the video stream
    int camera_device = parser.get<int>("camera");
    VideoCapture capture;
    capture.open(camera_device);
    if (!capture.isOpened())
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }
    Mat frame;

    // resize image capture
    int width = 640;
    int height = 480;
    capture.set(CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CAP_PROP_FRAME_HEIGHT, 480);
    // Mat image;

    // Capture frame each time
    while(capture.read(frame))
    {
        if(frame.empty())
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }

        // Apply the classifier to the frame
        detectAndDisplay(frame);
        if(waitKey(10) == 27)
        {
            break; // escape
        }
    }
    return 0;
}

void detectAndDisplay(Mat frame)
{
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    //-- Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);
    for( size_t i = 0; i < faces.size(); i++ )
   {
      // cout << faces.size() << endl;
      Mat faceROI = frame_gray( faces[i] );
      int x = faces[i].x;
      int y = faces[i].y;
      int h = y+faces[i].height;
      int w = x+faces[i].width;
      rectangle(frame,
          Point (x,y),
          Point (w,h),
          Scalar(255,0,255),
          2,
          8,
          0);
    }

    //-- Show what you got
    imshow("Capture - Face detection", frame);
}