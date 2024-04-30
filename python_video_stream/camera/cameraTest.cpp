/*
g++ -Wall -I /usr/local/include/i3system/ `pkg-config --cflags opencv` -L/usr/local/lib/ cameraTest.cpp `pkg-config --libs opencv` -li3system_usb_32 -li3system_te_32 -lusb-1.0 -o cameraTest
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/lib
g++ `pkg-config --cflags opencv` openCVTest.cpp `pkg-config --libs opencv` -o openCVTest
*/

#include <iostream>
#include "i3system_TE.h"
#include <opencv2/opencv.hpp>
#include <string> 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <stdio.h>
#include <opencv2/imgproc.hpp>

#define DEFAULT_DELTA 0.0

using namespace i3;
using namespace std;
using namespace cv;

void Hotplug_Callback(TE_STATE _teState);

/*
void createAlphaMat(Mat &mat)
{
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            Vec4b& rgba = mat.at<Vec4b>(i, j);
            rgba[0] = UCHAR_MAX;
            rgba[1] = saturate_cast<uchar>((float (mat.cols - j)) / ((float)mat.cols) * UCHAR_MAX);
            rgba[2] = saturate_cast<uchar>((float (mat.rows - i)) / ((float)mat.rows) * UCHAR_MAX);
            rgba[3] = saturate_cast<uchar>(0.5 * (rgba[1] + rgba[2]));
        }
    }
}
*/
int main()
{
    
    //Mat img;
    //Mat img8bit;
    double g_scalef= 1.0/256.0; // for scaling from 16 bit to 8 bit
    double g_deltaf=DEFAULT_DELTA;
    
    /*
    unsigned short pRecvImage[384*288];
    std::string s;
    std::string s1 = "images/img_";
    std::string s2 = ".jpg"; 
    std::stringstream ss;
    */

     

    
    // Set Hotplug Callback Function
    SetHotplugCallback(Hotplug_Callback);

    // Scan connected TE.
    SCANINFO *pScan = new SCANINFO[MAX_USB_NUM];
    ScanTE(pScan);
    int hnd_dev = -1;
    for(int i = 0; i < MAX_USB_NUM; i++){
        if(pScan[i].bDevCon){
            hnd_dev = i;
            break;
        }
    }
    delete pScan;

    // Open Connected Device
    if(hnd_dev != -1){
        TE_B *pTE = OpenTE_B(I3_TE_V1,hnd_dev);
        bool exit = false;
        if(pTE){
            cout << "TE Opened" << endl;

            // Read Flash Data
            if(pTE->ReadFlashData() == 1){
                cout << "Read Flash Data" << endl;

                int width = pTE->GetImageWidth(), height = pTE->GetImageHeight();
                unsigned short *pImgBuf = new unsigned short[width*height];

                cout<<"Image heigh : "<<height<<" and width : "<<width<<endl;

                for(int i = 0; i < 1; ++i){

                    // Get Image Data
                    if(pTE->RecvImage(pImgBuf)){
                        cout << "Image Received" << endl;

                        Mat img = Mat::zeros( width, height, CV_16U );
                        Mat img8bit = Mat::zeros( width, height, CV_8U );
                        img = Mat(width, height, CV_16U, pImgBuf);
                        img.convertTo(img8bit, CV_8U, g_scalef, g_deltaf);
                        imwrite("alpha.png", img8bit);
                        

                        /*
                        // Create mat with alpha channel
                        Mat mat(480, 640, CV_8UC4);
                        createAlphaMat(mat);
                        imshow("image", mat);
                        */
                        
                        /*
                        vector<int> compression_params;
                        compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
                        compression_params.push_back(9);

                        try {
                                imwrite("alpha.png", mat, compression_params);
                            }
                        catch (runtime_error& ex) 
                        {
                            fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
                            return 1;
                        }
                        */

                        // Get Tamperature at (x, y)
                        int x = 100, y = 100;
                        float temp = pTE->CalcTemp(x, y);
                        cout << "Temperature = " << temp << endl;
                    }
                    else{
                        cout << "Image not received" << endl;
                    }
                }
                delete pImgBuf;
            }
            else{
                cout << "Fail to Read Flash Data" << endl;
            }

            // Close Device
            pTE->CloseTE();
            cout << "Close Usb" << endl;

        }
        else{
            cout << "Open Failed" << endl;
        }
    }
    else{
        cout << "Device Not Connected" << endl;
    }
    return 0;
}

// Callback function executed when TE is arrived to or removed from usb port.
void Hotplug_Callback(TE_STATE _teState){
    if(_teState.nUsbState == TE_ARRIVAL){
        // Do something ...
    }
    else if(_teState.nUsbState == TE_REMOVAL){
        // Do something ...
    }
}
