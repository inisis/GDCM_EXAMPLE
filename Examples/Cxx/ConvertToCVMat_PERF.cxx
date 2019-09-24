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

bool percentile(cv::Mat& imageCv, std::vector<float>& query, std::vector<float>& answer)
{
//    type
//              C1	C2	C3	C4
//    CV_8U	    0	8	16	24
//    CV_8S	    1	9	17	25
//    CV_16U	2	10	18	26
//    CV_16S	3	11	19	27
//    CV_32S	4	12	20	28
//    CV_32F	5	13	21	29
//    CV_64F	6	14	22	30

//    depth
//    #define CV_8U   0
//    #define CV_8S   1
//    #define CV_16U  2
//    #define CV_16S  3
//    #define CV_32S  4
//    #define CV_32F  5
//    #define CV_64F  6

    auto depth = imageCv.depth();
    int upper = 0;
    int lower = 0;

    if(depth == 0){
        upper =  256;
        lower = 0;
    } else if(depth == 1){
        upper =  128;
        lower = -128;
    } else if(depth == 2){
        upper = 65536;
        lower = 0;
    } else if(depth == 3){
        upper = 32768;
        lower = -32768;
    } else{
        LOG(ERROR) << "Unsupported image depth" << depth;
        return -1;
    }

    float range[]={(float)lower, (float)upper /*exclusive*/};
    const float* ranges[] = {range};

    int channel[] = {0};
    const int* channels = {channel};

    int histsize[] = {upper-lower};
    const int* histSize = {histsize};

    cv::Mat hist;
    cv::calcHist(&imageCv, 1, channels, cv::Mat(), hist, 1, histSize, ranges);

    for(int i = 0; i < query.size(); ++i){
        int lowerbound = 0;
        for (float count=0.0; lowerbound < upper; lowerbound++) {
            // number of pixel at imin and below must be > number
            if ((count+=hist.at<float>(lowerbound)) >= imageCv.total() * query[i]){
                answer.push_back(lowerbound);
                break;
            }
        }
    }
}

bool convert_image_to_cv_grey(gdcm::Image const & gimage, cv::Mat &imageCv){
    const unsigned int* dimension = gimage.GetDimensions();

    unsigned int dimX = dimension[0];
    unsigned int dimY = dimension[1];

    std::vector<char> vbuffer;
    vbuffer.resize( gimage.GetBufferLength() );
    char *buffer = &vbuffer[0];
    gimage.GetBuffer(buffer);

    if(gimage.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::RGB){
        if(gimage.GetPixelFormat() != gdcm::PixelFormat::UINT8){
            return 1;
        }
        auto *ubuffer = (unsigned char*)buffer;
        imageCv = cv::Mat(dimY, dimX, CV_8UC3, ubuffer);
    } else if( gimage.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::MONOCHROME2 || gimage.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::MONOCHROME1){
        if(gimage.GetPixelFormat() == gdcm::PixelFormat::UINT8){
            // We need to copy each individual 8bits into R / G and B:
            auto *ubuffer = new unsigned char[dimX*dimY*3];
            for(unsigned int i = 0; i < dimX*dimY; i++){
                *ubuffer++ = *buffer;
                *ubuffer++ = *buffer;
                *ubuffer++ = *buffer++;
            }
      
            imageCv = cv::Mat(dimY, dimX, CV_8UC3, ubuffer);
        } else if(gimage.GetPixelFormat() == gdcm::PixelFormat::UINT16){
            auto *ubuffer16 = (unsigned short*)buffer;

            cv::Mat tmp = cv::Mat(dimY, dimX, CV_16UC1, ubuffer16);
            cv::Mat u16_(dimY, dimX, CV_16UC1);

            if(gimage.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::MONOCHROME1){
                cv::bitwise_not(tmp, u16_);
            }
            else{
                u16_ = tmp;
            }

            std::vector<float> query{0.01, 0.99};
            std::vector<float> answer;
            percentile(u16_, query, answer);

            cv::Mat before_eh(dimY, dimX, CV_8UC1);

            auto *u16_ptr = u16_.ptr<unsigned short>(0);

            auto *before_eh_ptr = before_eh.ptr<unsigned char>(0);

            for(unsigned int i = 0; i < dimX*dimY; i++){
                unsigned short x = clip(u16_ptr[i], answer[0], answer[1]);
                *before_eh_ptr++ = (unsigned char)((x - answer[0]) * 255.0 / (answer[1]  - answer[0]));
            }

            imageCv = cv::Mat(dimY, dimX, CV_8UC1);
            cv::equalizeHist(before_eh, imageCv);

        } else{
            std::cerr << "Pixel Format is: " << gimage.GetPixelFormat() << std::endl;
            return 1;
        }
    } else{
        std::cerr << "Unhandled PhotometricInterpretation: " << gimage.GetPhotometricInterpretation() << std::endl;
        return 1;
    }

    return 0;
}

template<typename T>
void check_exact_result(const T* const ref, const T* const gpu, size_t numElem) {
    bool is_same = true;
    for (size_t i = 0; i < numElem; ++i) {
        if (ref[i] != gpu[i]) {
            LOG(INFO) << "Difference at pos " << i;
            LOG(INFO) << "Reference: " << std::setprecision(17) << +ref[i] << "\n GPU      : " << +gpu[i];
            is_same = false;
        }
    }
    if(is_same)
        LOG(INFO) << "Generated images are the same.";
    else
        LOG(INFO) << "Generated images are not the same.";
}

int main(int argc, char *argv[])
{
    if( argc < 2 ){
        LOG(ERROR) << "Missing input parameter...";
        return 1;
    }

    const char *filename = argv[1];
    const char *outfilename = argv[2];

    std::string uri ="https://sc.jfhealthcare.cn/v1/picl/aets/piclarc/wado?requestType=WADO&contentType=application/dicom&studyUID=1.2.840.473.8013.20190624.1134240.765.29631.53&seriesUID=1.2.392.200036.9125.3.1045202532727.64910469116.4200806&objectUID=1.2.392.200036.9125.4.0.470808524.687605096.902437659";
    std::vector<char> data;

    LOG(INFO) << "Before read...";

    int result = read_image_from_uri(uri, data, 3000);

    LOG(INFO) << "Download from network: " << data.size();

    std::string s((char*)data.data(), data.size());
    std::istringstream iss(s);
    gdcm::ImageReader ir_buffer;
    ir_buffer.SetStream(iss);

    if(!ir_buffer.Read()){
        LOG(ERROR) << "Read Image Failed...";
        return 1;
    }

    const gdcm::Image &image_buffer = ir_buffer.GetImage();

    LOG(INFO) << "Read from buffer length: " << image_buffer.GetBufferLength();

    cv::Mat grey_image_buffer;
    if(convert_image_to_cv_grey(image_buffer, grey_image_buffer) == 1){
        LOG(ERROR) << "Convert Image Failed...";
        return 1;
    }
    LOG(INFO) << "Convert Image Done...";

    const cv::Size newSize(1024, 1024);
    cv::Mat output;
    cv::resize(grey_image_buffer, output, newSize);

    LOG(INFO) << "Resize Image Done...";

    cv::imwrite(outfilename, grey_image_buffer);

    return 0;
}