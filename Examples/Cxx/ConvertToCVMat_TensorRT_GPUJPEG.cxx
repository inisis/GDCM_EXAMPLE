#include "iostream"
#include "algorithm"
#include "map"
#include "gdcmImageReader.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <curl/curl.h>
#include <glog/logging.h>
#include <gdcmDecoder.h>

#include <cuda_runtime_api.h>

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"

#include "Lion.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

#define clip(x, a, b) x >= a ? (x < b ? x : b-1) : a;

#define CHECK_CUDA(status)                              \
{                                                       \
    do {cudaError_t eCUDAResult;                        \
        eCUDAResult = status;                           \
        if (eCUDAResult != cudaSuccess) {               \
        LOG(ERROR)<<" cuda exec faild ! "               \
        << " err_msg:" << cudaGetErrorString(status)    \
        << "; error_code:" << (status)                  \
        << "; code:" << (#status)                         \
        << __FILE__                                     \
        << __LINE__;                                    \
        abort();                                        \
    }}while(0);                                         \
}

class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kERROR)
        : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cerr << "INFO: "; break;
        default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};

static Logger gLogger;

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
//    CV_8U	0	8	16	24
//    CV_8S	1	9	17	25
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
    vbuffer.resize(gimage.GetBufferLength());
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

void doInference(IExecutionContext& context, const std::string& input_blob, const std::vector<std::string>& output_blob, float* inputData, int batchSize, std::vector<std::map<std::string, std::vector<float>>> &results)
{
    const ICudaEngine& engine = context.getEngine();
    assert(engine.getNbBindings() == (1 + output_blob.size()));
    void* buffers[1 + output_blob.size()] = {nullptr};

    int inputIndex = engine.getBindingIndex(input_blob.c_str());
    if(inputIndex < 0){
        LOG(ERROR) << "No Blob: " << input_blob;
    }

    DimsNCHW dims = static_cast<DimsNCHW &&>(engine.getBindingDimensions(inputIndex));

    LOG(INFO) << "Input C H W: " << dims.c() << " " << dims.h() << " " << dims.w();

    CHECK_CUDA(cudaMalloc(&buffers[inputIndex], batchSize *  dims.c() * dims.h() * dims.w() * sizeof(float))); // Data

    for (int i = 0; i < output_blob.size(); i++) {
        int index = engine.getBindingIndex(output_blob[i].c_str());
	if(index < 0){
	LOG(ERROR) << "No Blob: " << output_blob[i];
	}
        DimsNCHW dims = static_cast<DimsNCHW &&>(engine.getBindingDimensions(index));
        LOG(INFO) << "Output C H W: " << dims.c() << " " << dims.h() << " " << dims.w();
	
        cudaMalloc(&buffers[index], batchSize * dims.c() * dims.h() * dims.w() * sizeof(float));
    }

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    CHECK_CUDA(cudaMemcpyAsync(buffers[inputIndex], inputData, batchSize * dims.c() * dims.h() * dims.w() * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);

    results.resize(batchSize);
    for(int i = 0; i < output_blob.size(); i++)
    {
	std::vector<float> output;
	std::string layername = output_blob[i];
        auto index = engine.getBindingIndex(layername.c_str());
        DimsNCHW dims = static_cast<DimsNCHW &&>(engine.getBindingDimensions(index));
        auto fea_len =  dims.c() * dims.h() * dims.w();
        output.resize(batchSize * fea_len);
        cudaMemcpyAsync(output.data(), buffers[index], batchSize * fea_len * sizeof(float), cudaMemcpyDeviceToHost, stream);

        for(int j = 0; j < batchSize; j++)
        {
	    std::map<std::string, std::vector<float>> &feature = results[j];
            feature[layername].resize(fea_len);
            std::copy(output.begin() + j * fea_len, output.begin() + (j + 1) * fea_len, feature[layername].begin());
        }
    }

    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    for (int i = 0; i < engine.getNbBindings(); ++i) {
        if (buffers[i] != nullptr) {
            CHECK_CUDA(cudaFree(buffers[i]));
        }
    }
}

int main(int argc, char *argv[])
{
    if( argc < 2 ){
        LOG(ERROR) << "Missing input parameter...";
        return 1;
    }

    const char *filename = argv[1];
    const char *outfilename = argv[2];
    int gpuid = atoi(argv[3]);

    std::string uri ="https://sc.jfhealthcare.cn/v1/picl/aets/piclarc/wado?requestType=WADO&contentType=application/dicom&studyUID=1.2.840.473.8013.20190624.1134240.765.29631.53&seriesUID=1.2.392.200036.9125.3.1045202532727.64910469116.4200806&objectUID=1.2.392.200036.9125.4.0.470808524.687605096.902437659";
    std::vector<char> uri_data;

    LOG(INFO) << "Before read...";

    int result = read_image_from_uri(uri, uri_data, 5000);

    LOG(INFO) << "Download from network: " << uri_data.size();

    std::string s((char*)uri_data.data(), uri_data.size());
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
    cv::imwrite(outfilename, grey_image_buffer);

    cudaSetDevice(gpuid);
    int batch_size = 1;
    int input_channle = 3;
    int input_height = 1024;
    int input_width = 1024;
    float pixel_mean = 128;

    const unsigned int* dimension = image_buffer.GetDimensions();
    unsigned int Width = dimension[0];
    unsigned int Height = dimension[1];

    int new_width = 0;
    int new_height = 0;

    if(Width > Height){
        auto ratio = Width * 1.0 / Height;
	new_width = input_width;
        new_height = round(new_width / ratio);
    }
    else{
        auto ratio = Height * 1.0 / Width;
        new_height = input_height;
        new_width = round(new_height / ratio);
    }
    LOG(INFO) << "new_widht, new_height: " << new_width << " " << new_height;

    const cv::Size newSize(new_width, new_height);
    cv::Mat output;
    cv::resize(grey_image_buffer, output, newSize);

    cv::Mat paddinged_mat(input_height, input_width, CV_8UC1, cv::Scalar::all(pixel_mean));
    cv::Mat srcROI(paddinged_mat, cv::Rect(0, 0, output.cols, output.rows));
    output.copyTo(srcROI);
    
    LOG(INFO) << "Resize Image Done...";
    
    std::ifstream cached_model;
    cached_model.open("tb_FP32_1_61.dat", std::ios::binary);
    if(!cached_model){
        LOG(FATAL) << " falied to open file " << "tb_FP32_1_61.dat";
    }
    IRuntime* runtime = createInferRuntime(gLogger);
    cached_model.seekg (0, cached_model.end);
    int length = cached_model.tellg();
    cached_model.seekg (0, cached_model.beg);

    char * buffer = new char [length];
    cached_model.read(buffer,length);
    ICudaEngine* engine = runtime->deserializeCudaEngine(buffer, length, nullptr);
    delete[] buffer;
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    std::string input_blob = "data";
    std::vector<std::string> output_blobs{"ClassifiernConv2dnconvn192", "ClassifiernLinearnfcn0n201"};
     
    std::vector<std::map<std::string, std::vector<float>>> results;
    float* data = new float[batch_size * input_channle * input_height * input_width];
    auto *reference_ptr = paddinged_mat.ptr<unsigned char>(0);

    for (int c = 0; c < input_channle; ++c)
        for (unsigned j = 0, volChl = input_height * input_width; j < volChl; ++j){
            data[c * volChl + j] = (float)(reference_ptr[j] - pixel_mean);
        }
    
    LOG(INFO) << "Before inference...";
    doInference(*context, input_blob, output_blobs, data, batch_size, results);

    for (int idx = 0; idx < results.size(); idx++){
        LOG(INFO) << "Confidence: " << results[idx]["ClassifiernLinearnfcn0n201"][0];
        LOG(INFO) << "Featuremap: " << results[idx]["ClassifiernConv2dnconvn192"].size();
    }

    unsigned char* feat_map = new unsigned char[results[0]["ClassifiernConv2dnconvn192"].size()];

    for(int i = 0; i < results[0]["ClassifiernConv2dnconvn192"].size(); ++i){
        data[i] = results[0]["ClassifiernConv2dnconvn192"][i] * 255;
    }

    auto long_side = (int)std::max(Width, Height);
    cv::Mat feat_map_mat = cv::Mat(32, 32, CV_8UC1, feat_map);
    cv::Mat feat_map_mat_long_side;
    cv::resize(feat_map_mat, feat_map_mat_long_side, cv::Size{long_side, long_side});
    cv::Mat feat_map_mat_orig_size(feat_map_mat_long_side, cv::Rect(0, 0, Width, Height));
    
    cv::Mat img_color;
    cv::applyColorMap(feat_map_mat_orig_size, img_color, cv::COLORMAP_JET);

    LOG(INFO) << "Channel: " << img_color.channels();
    LOG(INFO) << "Channel: " << feat_map_mat_orig_size.channels();
    LOG(INFO) << "Channel: " << feat_map_mat_long_side.channels();
    
    LOG(INFO) << "Width: " << img_color.cols;
    LOG(INFO) << "Height: " << img_color.rows;

    auto *gpujpeg = new FrozenThrone::Lion();
    
    gpujpeg->init_encoder(gpuid, 4000, 4000, 1);

    cv::Size size = {img_color.cols, img_color.rows};

    void *bgr_data;
    cudaMalloc((void **) &bgr_data, size.width * size.height * 3);
    cudaMemcpy(bgr_data, img_color.ptr<unsigned char>(0), size.width * size.height * 3, cudaMemcpyHostToDevice);

    uint8_t *image_compressed = nullptr;
    int image_compressed_size = 0;

    gpujpeg->encode_bgr((uint8_t *)bgr_data, size, image_compressed, image_compressed_size);
    
    FILE* file = fopen("output_gpu.jpg", "wb");
    if ( !file ) {
        LOG(ERROR) << "[GPUJPEG] [Error] Failed open output_gpu.jpg for writing!";
        return -1;
    }

    if (image_compressed_size != fwrite(image_compressed, sizeof(uint8_t), image_compressed_size, file)) {
        LOG(ERROR) << "[GPUJPEG] [Error] Failed to write image data [" << image_compressed_size << " bytes] to file output_gpu.jpg!";
        return -1;
    }
    fclose(file);

    cv::imwrite(outfilename, img_color);

    return 0;
}
