/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*
 * This example shows how to setup the pipeline from a gdcm::ImageReader into a
 * Qt QImage data structure.
 * It only handles 2D image.
 *
 * Ref:
 * http://doc.trolltech.com/4.5/qimage.html
 *
 * Usage:
 *  ConvertToQImage gdcmData/012345.002.050.dcm output.png

 * Thanks:
 *   Sylvain ADAM (sylvain51 hotmail com) for contributing this example
 */

#include "iostream"
#include "algorithm"
#include "gdcmImageReader.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <curl/curl.h>
#include <glog/logging.h>
#include <gdcmDecoder.h>

#define clip(x, a, b) x >= a ? (x < b ? x : b-1) : a;

static size_t image_write_callback(void *buffer, size_t size, size_t nmemb, void *stream)
{
    std::vector<char> *vec_buffer = (std::vector<char>*)stream;
    char *tmp = (char *)buffer;
    int length = (size * nmemb) / sizeof(char);
    std::copy(tmp, tmp + length, std::back_inserter(*vec_buffer));
    return size * nmemb;
}

static int read_image_from_uri(const std::string &uri, std::vector<char> &buffer, long timeout_ms)
{
    try
    {
        CURL *curl_handle = curl_easy_init();
        if (curl_handle == nullptr)
        {
            LOG(WARNING) << "ImageParser: curl_easy_init() failed! uri " << uri << std::endl;
            return 0;
        }

        // trim the spaces before or after the string.
        std::string uri_copy = uri;

        curl_easy_setopt(curl_handle, CURLOPT_URL, uri_copy.c_str());
        curl_easy_setopt(curl_handle, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl_handle, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl_handle, CURLOPT_SSL_VERIFYHOST, 0L);
        curl_easy_setopt(curl_handle, CURLOPT_VERBOSE, 0L); //0L for no verbose
        curl_easy_setopt(curl_handle, CURLOPT_NOPROGRESS, 1L); //1L for no progress
        curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, image_write_callback);
        curl_easy_setopt(curl_handle, CURLOPT_TIMEOUT_MS, timeout_ms);
        curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, &buffer);
        CURLcode res = curl_easy_perform(curl_handle);

        curl_easy_cleanup(curl_handle);

        if (CURLE_OK != res)
        {
            LOG(WARNING) << "ImageParser: [@@@@@@@@@]curl perform failed: " << res << ", uri " << uri << std::endl;
            return 0;
        }

        if (buffer.empty())
        {
            LOG(WARNING) << "ImageParser: [@@@@@@@@@]curl find empty image! uri " << uri << std::endl;
            return 0;
        }
    }
    catch (...)
    {
        LOG(WARNING) << "ImageParser: curl perform occured exception! uri " << uri << std::endl;
        buffer.clear();
        return 0;
    }

    return 0;
}


bool ConvertToFormat_GREY(gdcm::Image const & gimage, char *buffer, cv::Mat &imageCv)
{
    const unsigned int* dimension = gimage.GetDimensions();

    unsigned int dimX = dimension[0];
    unsigned int dimY = dimension[1];

    gimage.GetBuffer(buffer);

    // Let's start with the easy case:
    if( gimage.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::RGB )
    {
        if( gimage.GetPixelFormat() != gdcm::PixelFormat::UINT8 )
        {
            return false;
        }
        unsigned char *ubuffer = (unsigned char*)buffer;
        imageCv = cv::Mat(dimY, dimX, CV_8UC3, ubuffer);
    }
    else if( gimage.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::MONOCHROME2 || gimage.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::MONOCHROME1)
    {
        if( gimage.GetPixelFormat() == gdcm::PixelFormat::UINT8 )
        {
            // We need to copy each individual 8bits into R / G and B:
            unsigned char *ubuffer = new unsigned char[dimX*dimY*3];
            unsigned char *pubuffer = ubuffer;
            for(unsigned int i = 0; i < dimX*dimY; i++)
            {
                *pubuffer++ = *buffer;
                *pubuffer++ = *buffer;
                *pubuffer++ = *buffer++;
            }
      
            imageCv = cv::Mat(dimY, dimX, CV_8UC3, ubuffer);
        }
        else if( gimage.GetPixelFormat() == gdcm::PixelFormat::UINT16 )
        {
            // We need to copy each individual 16bits into R / G and B (truncate value)
            unsigned short *ubuffer16 = (unsigned short*)buffer;
            unsigned char *ubuffer = new unsigned char[dimX*dimY];
            unsigned char *pubuffer = ubuffer;

            cv::Mat tmp = cv::Mat(dimY, dimX, CV_16UC1, ubuffer16);
            cv::Mat u16_(dimY, dimX, CV_16UC1);
            if(gimage.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::MONOCHROME1)
            {
                cv::bitwise_not(tmp, u16_);
            }
            else
            {
                u16_ = tmp;
            }

            cv::Mat hist;
            
            float range[]={0.0f, 65536.0f};
            const float* ranges[] = {range};

            int channel[] = {0};
            const int* channels = {channel};

            int histsize[] = {65536};
            const int* histSize = {histsize};

            cv::calcHist(&u16_, 1, channels, cv::Mat(), hist, 1, histSize, ranges);

            float number= u16_.total()*0.01;

            int lowerbound = 0;
            for (float count=0.0; lowerbound < 65535; lowerbound++) {
                // number of pixel at imin and below must be > number
                if ((count+=hist.at<float>(lowerbound)) >= number)
                    break;
            }

            int upperbound = 65535;
            for (float count=0.0; upperbound >= 0; upperbound--) {
                // number of pixel at imax and below must be > number
                if ((count += hist.at<float>(upperbound)) >= number)
                    break;
            }

            std::cout<< lowerbound << " " << upperbound <<std::endl;

            unsigned short *outputPtr = u16_.ptr<unsigned short>(0);
            for(unsigned int i = 0; i < dimX*dimY; i++)
            {
                unsigned short x = clip(outputPtr[i], lowerbound, upperbound);
                *pubuffer++ = (unsigned char)((x - lowerbound) * 255.0 / (upperbound  - lowerbound));
            }
            cv::Mat* before_eh = new cv::Mat(dimY, dimX, CV_8UC1, ubuffer);
            imageCv = cv::Mat(dimY, dimX, CV_8UC1);
            cv::equalizeHist(*before_eh, imageCv);

        }
        else
        {
            std::cerr << "Pixel Format is: " << gimage.GetPixelFormat() << std::endl;
            return false;
        }
    }
    else
    {
        std::cerr << "Unhandled PhotometricInterpretation: " << gimage.GetPhotometricInterpretation() << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char *argv[])
{
    if( argc < 2 )
    {
        return 1;
    }
    const char *filename = argv[1];
    const char *outfilename = argv[2];

    gdcm::ImageReader ir_file;
    gdcm::ImageReader ir;
    ir_file.SetFileName( filename );
    if(!ir_file.Read())
    {
        //Read failed
        return 1;
    }

    std::string uri ="https://sc.jfhealthcare.cn/v1/picl/aets/piclarc/wado?requestType=WADO&contentType=application/dicom&studyUID=1.2.840.473.8013.20190624.1134240.765.29631.53&seriesUID=1.2.392.200036.9125.3.1045202532727.64910469116.4200806&objectUID=1.2.392.200036.9125.4.0.470808524.687605096.902437659";
    std::vector<char> data;
    int result = read_image_from_uri(uri, data, 5000);
    LOG(INFO) << data.size();

    std::string s((char*)data.data(),data.size());
    std::istringstream iss(s);
    ir.SetStream(iss);
    if(!ir.Read())
    {
        //Read failed
        return 1;
    }

    // const gdcm::Image &gimage = (gdcm::Image)data.data();
    LOG(INFO) << "Getting image from ImageReader...";

    // gdcm::Image* buffer_image = reinterpret_cast<gdcm::Image*>(data.data());

    //LOG(INFO) << buffer_image->GetBufferLength();

    const gdcm::Image &gimage = ir.GetImage();

    LOG(INFO) << gimage.GetBufferLength();

    const gdcm::Image &gimage_file = ir_file.GetImage();

    LOG(INFO) << gimage_file.GetBufferLength();

    std::vector<char> vbuffer;
    vbuffer.resize( gimage.GetBufferLength() );
    char *buffer = &vbuffer[0];

    cv::Mat imageCv;
    if( !ConvertToFormat_GREY( gimage, buffer, imageCv) )
    {
        return 1;
    }

    cv::imwrite(outfilename, imageCv);
    return 0;
}