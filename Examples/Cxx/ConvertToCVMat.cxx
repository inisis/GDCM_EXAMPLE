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

#include "algorithm"
#include "gdcmImageReader.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#define clip(x, a, b) x >= a ? (x < b ? x : b-1) : a;

bool ConvertToFormat_RGB888(gdcm::Image const & gimage, char *buffer, cv::Mat* &imageCv)
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
        imageCv = new cv::Mat(dimY, dimX, CV_8UC3, ubuffer);
    }
    else if( gimage.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::MONOCHROME2 )
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
      
            imageCv = new cv::Mat(dimY, dimX, CV_8UC3, ubuffer);
        }
        else if( gimage.GetPixelFormat() == gdcm::PixelFormat::UINT16 )
        {
            // We need to copy each individual 16bits into R / G and B (truncate value)
            unsigned short *ubuffer16 = (unsigned short*)buffer;
            unsigned char *ubuffer = new unsigned char[dimX*dimY];
            unsigned char *pubuffer = ubuffer;

            cv::Mat u16_ = cv::Mat(dimY, dimX, CV_16UC1, ubuffer16);

            cv::Mat hist;
            
            float range[]={0.0f, 65535.0f};
            const float* ranges[] = {range};

            int channel[] = {0};
            const int* channels = {channel};

            int histsize[] = {65535};
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

            for(unsigned int i = 0; i < dimX*dimY; i++)
            {
                unsigned short x = clip(*ubuffer16, lowerbound, upperbound);
                *pubuffer++ = (unsigned char)((x - lowerbound) * 255.0 / (upperbound  - lowerbound));
                ubuffer16++;
            }
            cv::Mat* before_eh = new cv::Mat(dimY, dimX, CV_8UC1, ubuffer);
            imageCv = new cv::Mat(dimY, dimX, CV_8UC1);
            cv::equalizeHist(*before_eh, *imageCv);

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

    gdcm::ImageReader ir;
    ir.SetFileName( filename );
    if(!ir.Read())
    {
        //Read failed
        return 1;
    }

    std::cout<<"Getting image from ImageReader..."<<std::endl;

    const gdcm::Image &gimage = ir.GetImage();
    std::vector<char> vbuffer;
    vbuffer.resize( gimage.GetBufferLength() );
    char *buffer = &vbuffer[0];

    cv::Mat* imageCv = NULL;
    if( !ConvertToFormat_RGB888( gimage, buffer, imageCv) )
    {
        return 1;
    }

    cv::imwrite(outfilename, *imageCv);
    return 0;
}