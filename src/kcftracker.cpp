/*

Tracker based on Kernelized Correlation Filter (KCF) [1] and Circulant Structure with Kernels (CSK) [2].
CSK is implemented by using raw gray level features, since it is a single-channel filter.
KCF is implemented by using HOG features (the default), since it extends CSK to multiple channels.

[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

[2] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"Exploiting the Circulant Structure of Tracking-by-detection with Kernels", ECCV 2012.

Authors: Joao Faro, Christian Bailer, Joao F. Henriques
Contacts: joaopfaro@gmail.com, Christian.Bailer@dfki.de, henriques@isr.uc.pt
Institute of Systems and Robotics - University of Coimbra / Department Augmented Vision DFKI


Constructor parameters, all boolean:
    hog: use HOG features (default), otherwise use raw pixels
    fixed_window: fix window size (default), otherwise use ROI size (slower but more accurate)
    multiscale: use multi-scale tracking (default; cannot be used with fixed_window = true)

Default values are set for all properties of the tracker depending on the above choices.
Their values can be customized further before calling init():
    interp_factor: linear interpolation factor for adaptation
    sigma: gaussian kernel bandwidth
    lambda: regularization
    cell_size: HOG cell size
    padding: horizontal area surrounding the target, relative to its size
    output_sigma_factor: bandwidth of gaussian target
    template_size: template size in pixels, 0 to use ROI size
    scale_step: scale step for multi-scale estimation, 1 to disable it
    scale_weight: to downweight detection scores of other scales for added stability

For speed, the value (template_size/cell_size) should be a power of 2 or a product of small prime numbers.

Inputs to init():
   image is the initial frame.
   roi is a cv::Rect with the target positions in the initial frame

Inputs to update():
   image is the current frame.

Outputs of update():
   cv::Rect with target positions for the current frame


By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
 */

#ifndef _KCFTRACKER_HEADERS
#include "kcftracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "labdata.hpp"
#endif

#include "utils.hpp"
#include "threadpool.hpp"

#include <corecrt_math_defines.h>
#include <iostream>
#include <chrono>


//#define TIME_TEST   1


using std::cout;
using std::endl;

//#define M_PI 3.14159265358979323846


class KCFTracker::MultiThreadHelper
{
public:
    fixed_thread_pool thread_pool;
    float peak_values[3];
    cv::Point2f res_pos[3];
    int processed_cnt;
    std::mutex mtx;
    std::condition_variable cond;

    MultiThreadHelper() 
        : thread_pool(2)
        , peak_values{}
        , res_pos{}
        , processed_cnt(0)
    {
    }
};


// Constructor
KCFTracker::KCFTracker(const bool& lab, const bool& multi_thread)
{
    // Parameters equal in all cases
    lambda = 0.0001;                // 正则项
    padding = 2.5;                  // 相对于尺寸的包围区域
    //output_sigma_factor = 0.1;
    output_sigma_factor = 0.125;    // 目标高斯带宽
    _labfeatures = lab;             // lab色空间特征

    // Set the parameters depending on the KCF mode
    setParameters();

    multi_thread_helper_ = multi_thread ? new MultiThreadHelper() : nullptr;
}

KCFTracker::~KCFTracker()
{
    delete multi_thread_helper_;
}

void KCFTracker::setParameters()
{
    // VOT dataset evaluation
    interp_factor = 0.012;  // 用于适应的线性插值因子
    interp_threshold = 0.4;
    sigma = 0.6;            // 高斯核带宽

    cell_size = 4;          // HOG格子尺寸

    if (_labfeatures) {
        interp_factor = 0.005;
        interp_threshold = 0.25;
        sigma = 0.4;
        //output_sigma_factor = 0.025;
        output_sigma_factor = 0.1;  // 目标高斯带宽

        _labCentroids = cv::Mat(nClusters, 3, CV_32FC1, &data); // lab
        cell_sizeQ = cell_size * cell_size;   // cell size^2, to avoid repeated operations
    }

    template_size = 96; // make it small
    scale_step = 1.05;
    scale_weight = 0.95;
}

// Initialize tracker
void KCFTracker::init(const cv::Rect &roi, const cv::Mat& image)
{
    _roi = roi;
    assert(roi.width >= 0 && roi.height >= 0);

    {
#ifdef TIME_TEST
        // 计时开始
        double ratio = (double)
            std::chrono::steady_clock::duration::period::num
            / std::chrono::steady_clock::duration::period::den;
        auto start = std::chrono::steady_clock::now();
#endif // TIME_TEST
        _tmpl = getFeatures(image, _tmplate_img[0], 1, _size_patch);
#ifdef TIME_TEST
        // 计时结束并打印计时
        auto end = std::chrono::steady_clock::now();
        cout << "Init getFeatures: "
            << (end - start).count() * ratio << endl;
#endif // TIME_TEST
    }

    {
#ifdef TIME_TEST
        // 计时开始
        double ratio = (double)
            std::chrono::steady_clock::duration::period::num
            / std::chrono::steady_clock::duration::period::den;
        auto start = std::chrono::steady_clock::now();
#endif // TIME_TEST
        _prob = createGaussianPeak(_size_patch[0], _size_patch[1]);
#ifdef TIME_TEST
        // 计时结束并打印计时
        auto end = std::chrono::steady_clock::now();
        cout << "Init createGaussianPeak: "
            << (end - start).count() * ratio << endl;
#endif // TIME_TEST

    }

    {
#ifdef TIME_TEST
        // 计时开始
        double ratio = (double)
            std::chrono::steady_clock::duration::period::num
            / std::chrono::steady_clock::duration::period::den;
        auto start = std::chrono::steady_clock::now();
#endif // TIME_TEST
        _alphaf = cv::Mat(_size_patch[0], _size_patch[1], CV_32FC2, float(0));
#ifdef TIME_TEST
        // 计时结束并打印计时
        auto end = std::chrono::steady_clock::now();
        cout << "Init _alphaf: "
            << (end - start).count() * ratio << endl;
#endif // TIME_TEST
    }

    {
#ifdef TIME_TEST
        // 计时开始
        double ratio = (double)
            std::chrono::steady_clock::duration::period::num
            / std::chrono::steady_clock::duration::period::den;
        auto start = std::chrono::steady_clock::now();
#endif // TIME_TEST
        train(_tmpl, 1.0, _size_patch); // train with initial frame
#ifdef TIME_TEST
        // 计时结束并打印计时
        auto end = std::chrono::steady_clock::now();
        cout << "Init train: "
            << (end - start).count() * ratio << endl;
#endif // TIME_TEST
    }
 }

// Update position based on the new frame
cv::Rect KCFTracker::update(const cv::Mat& image, float& prob)
{
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;

    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;


    float peak_value;
    cv::Point2f res;
    if (multi_thread_helper_)
    {
        // 多线程
        {
            std::unique_lock<std::mutex> lk(multi_thread_helper_->mtx);
            multi_thread_helper_->processed_cnt = 0;
            multi_thread_helper_->mtx.unlock();
            multi_thread_helper_->thread_pool.execute([this, &image] {
                thread_local static int size_patch[3];
                auto features = getFeatures(image, _tmplate_img[0], false, size_patch, 1.0f / scale_step);
                multi_thread_helper_->res_pos[1] = detect(_tmpl, features, multi_thread_helper_->peak_values[1], size_patch);
                multi_thread_helper_->mtx.lock();
                if ((++multi_thread_helper_->processed_cnt) >= 2)
                {
                    multi_thread_helper_->mtx.unlock();
                    multi_thread_helper_->cond.notify_all();
                }
                else
                {
                    multi_thread_helper_->mtx.unlock();
                }
            });
            multi_thread_helper_->thread_pool.execute([this, &image] {
                thread_local static int size_patch[3];
                auto features = getFeatures(image, _tmplate_img[1], false, size_patch, scale_step);
                multi_thread_helper_->res_pos[2] = detect(_tmpl, features, multi_thread_helper_->peak_values[2], size_patch);
                multi_thread_helper_->mtx.lock();
                if ((++multi_thread_helper_->processed_cnt) >= 2)
                {
                    multi_thread_helper_->mtx.unlock();
                    multi_thread_helper_->cond.notify_all();
                }
                else
                {
                    multi_thread_helper_->mtx.unlock();
                }
            });

            auto features = getFeatures(image, _tmplate_img[2], false, _size_patch, 1.0f);
            multi_thread_helper_->res_pos[0] = detect(_tmpl, features, multi_thread_helper_->peak_values[0], _size_patch);
            multi_thread_helper_->mtx.lock();
            if ((multi_thread_helper_->processed_cnt) < 2)
            {
                multi_thread_helper_->cond.wait(lk);
            }
        }

        peak_value = multi_thread_helper_->peak_values[0];
        res = multi_thread_helper_->res_pos[0];

        if (peak_value < scale_weight * multi_thread_helper_->peak_values[1])
        {
            res = multi_thread_helper_->res_pos[1];
            peak_value = multi_thread_helper_->peak_values[1];
            _scale /= scale_step;
            _roi.width /= scale_step;
            _roi.height /= scale_step;
        }
        if (peak_value < scale_weight * multi_thread_helper_->peak_values[2])
        {
            res = multi_thread_helper_->res_pos[2];
            peak_value = multi_thread_helper_->peak_values[2];
            _scale *= scale_step;
            _roi.width *= scale_step;
            _roi.height *= scale_step;
        }
    }
    else
    {
#ifdef TIME_TEST
        double ratio = (double)
            std::chrono::steady_clock::duration::period::num
            / std::chrono::steady_clock::duration::period::den;
        std::chrono::steady_clock::time_point start, end;
#endif // TIME_TEST
        {
#ifdef TIME_TEST
            start = std::chrono::steady_clock::now();
#endif // TIME_TEST
            auto features = getFeatures(image, _tmplate_img[0], 0, _size_patch, 1.0f);
#ifdef TIME_TEST
            end = std::chrono::steady_clock::now();
            cout << "Update getFeatures 1: "
                << (end - start).count() * ratio << endl;
#endif // TIME_TEST

#ifdef TIME_TEST
            start = std::chrono::steady_clock::now();
#endif // TIME_TEST
            res = detect(_tmpl, features, peak_value, _size_patch);
#ifdef TIME_TEST
            end = std::chrono::steady_clock::now();
            cout << "Update detect 1: "
                << (end - start).count() * ratio << endl;
#endif // TIME_TEST
        }

        if (scale_step != 1) {
            // Test at a smaller _scale
            float new_peak_value;

#ifdef TIME_TEST
            start = std::chrono::steady_clock::now();
#endif // TIME_TEST
            auto features = getFeatures(image, _tmplate_img[1], 0, _size_patch, 1.0f / scale_step);
#ifdef TIME_TEST
            end = std::chrono::steady_clock::now();
            cout << "Update getFeatures 2: "
                << (end - start).count() * ratio << endl;
#endif // TIME_TEST

#ifdef TIME_TEST
            start = std::chrono::steady_clock::now();
#endif // TIME_TEST
            cv::Point2f new_res = detect(_tmpl, features, new_peak_value, _size_patch);
#ifdef TIME_TEST
            end = std::chrono::steady_clock::now();
            cout << "Update detect 2: "
                << (end - start).count() * ratio << endl;
#endif // TIME_TEST

            if (scale_weight * new_peak_value > peak_value) {
                res = new_res;
                peak_value = new_peak_value;
                _scale /= scale_step;
                _roi.width /= scale_step;
                _roi.height /= scale_step;
            }

            // Test at a bigger _scale
#ifdef TIME_TEST
            start = std::chrono::steady_clock::now();
#endif // TIME_TEST
            features = getFeatures(image, _tmplate_img[2], 0, _size_patch, scale_step);
#ifdef TIME_TEST
            end = std::chrono::steady_clock::now();
            cout << "Update getFeatures 3: "
                << (end - start).count() * ratio << endl;
#endif // TIME_TEST

#ifdef TIME_TEST
            start = std::chrono::steady_clock::now();
#endif // TIME_TEST
            new_res = detect(_tmpl, features, new_peak_value, _size_patch);
#ifdef TIME_TEST
            end = std::chrono::steady_clock::now();
            cout << "Update detect 3: "
                << (end - start).count() * ratio << endl;
#endif // TIME_TEST

            if (scale_weight * new_peak_value > peak_value) {
                res = new_res;
                peak_value = new_peak_value;
                _scale *= scale_step;
                _roi.width *= scale_step;
                _roi.height *= scale_step;
            }
        }
    }

    prob = peak_value;

    // Adjust by cell size and _scale
    _roi.x = cx - _roi.width / 2.0f + ((float) res.x * cell_size * _scale);
    _roi.y = cy - _roi.height / 2.0f + ((float) res.y * cell_size * _scale);

    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

    assert(_roi.width >= 0 && _roi.height >= 0);

#ifdef TIME_TEST
    start = std::chrono::steady_clock::now();
#endif // TIME_TEST
    cv::Mat x = getFeatures(image, _tmplate_img[0], 0, _size_patch);
#ifdef TIME_TEST
    end = std::chrono::steady_clock::now();
    cout << "Update getFeatures x: "
        << (end - start).count() * ratio << endl;
#endif // TIME_TEST

#ifdef TIME_TEST
    start = std::chrono::steady_clock::now();
#endif // TIME_TEST
    if(prob > interp_threshold)
        train(x, interp_factor, _size_patch);
#ifdef TIME_TEST
    end = std::chrono::steady_clock::now();
    cout << "Update train x: "
        << (end - start).count() * ratio << endl;
#endif // TIME_TEST

    return _roi;
}

// Detect object in the current frame.
cv::Point2f KCFTracker::detect(const cv::Mat& z, const cv::Mat& x, float &peak_value, int* size_patch)
{
    using namespace FFTTools;

    cv::Mat k = gaussianCorrelation(x, z, size_patch);
    cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));

    //minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
    cv::Point2i pi;
    double pv;
    cv::minMaxLoc(res, NULL, &pv, NULL, &pi);
    peak_value = (float) pv;

    //subpixel peak estimation, coordinates will be non-integer
    cv::Point2f p((float)pi.x, (float)pi.y);

    if (pi.x > 0 && pi.x < res.cols-1) {
        p.x += subPixelPeak(res.at<float>(pi.y, pi.x-1), peak_value, res.at<float>(pi.y, pi.x+1));
    }

    if (pi.y > 0 && pi.y < res.rows-1) {
        p.y += subPixelPeak(res.at<float>(pi.y-1, pi.x), peak_value, res.at<float>(pi.y+1, pi.x));
    }

    p.x -= (res.cols) / 2;
    p.y -= (res.rows) / 2;

    return p;
}

// train tracker with a single image
void KCFTracker::train(const cv::Mat& x, const float& train_interp_factor, int* size_patch)
{
    using namespace FFTTools;

    const cv::Mat k = gaussianCorrelation(x, size_patch);
    const cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda));

    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
    _alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor) * alphaf;
}

// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
cv::Mat KCFTracker::gaussianCorrelation(const cv::Mat& x1, const cv::Mat& x2, int* size_patch)
{
    using namespace FFTTools;
    cv::Mat c = cv::Mat(cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0));
    // HOG features
    cv::Mat caux;
    cv::Mat x1aux;
    cv::Mat x2aux;
    for (int i = 0; i < size_patch[2]; i++) {
        x1aux = x1.row(i).reshape(1, size_patch[0]);   // Procedure do deal with cv::Mat multichannel bug
        x2aux = x2.row(i).reshape(1, size_patch[0]);
        cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
        caux = fftd(caux, true);
        addRearrangeReal_float(c, caux);
    }

    float x1_x2_pow2_sum = sumPow2_ch1(x1) + sumPow2_ch1(x2);
    float size_patch_den = 1.f / (size_patch[0] * size_patch[1] * size_patch[2]);
    float sigma_pow2_den = 1.f / (sigma * sigma);
    float* c_data = (float*)c.data;
    for (int i = 0; i < c.rows * c.cols; ++i)
    {
        float temp = x1_x2_pow2_sum - 2 * c_data[i];
        c_data[i] = (temp > 0.f) ? expf(-temp * size_patch_den * sigma_pow2_den) : 1.f;
    }
    return c;
}

cv::Mat KCFTracker::gaussianCorrelation(const cv::Mat& x, int* size_patch)
{
    using namespace FFTTools;
    cv::Mat c = cv::Mat(cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0));
    // HOG features
    cv::Mat caux;
    cv::Mat xaux;
    for (int i = 0; i < size_patch[2]; i++) {
        xaux = x.row(i).reshape(1, size_patch[0]);   // Procedure do deal with cv::Mat multichannel bug
        cv::Mat xaux_fftd = fftd(xaux);
        cv::mulSpectrums(xaux_fftd, xaux_fftd, caux, 0, true);
        caux = fftd(caux, true);
        addRearrangeReal_float(c, caux);
    }

    float x_x_pow2_sum = 2 * sumPow2_ch1(x);
    float size_patch_den = 1.f / (size_patch[0] * size_patch[1] * size_patch[2]);
    float sigma_pow2_den = 1.f / (sigma * sigma);
    float* c_data = (float*)c.data;
    for (int i = 0; i < c.rows * c.cols; ++i)
    {
        float temp = x_x_pow2_sum - 2 * c_data[i];
        c_data[i] = (temp > 0.f) ? expf(-temp * size_patch_den * sigma_pow2_den) : 1.f;
    }
    return c;
}


// Create Gaussian Peak. Function called only in the first frame.
cv::Mat KCFTracker::createGaussianPeak(const size_t& sizey, const size_t& sizex)
{
    cv::Mat_<float> res(sizey, sizex);

    const size_t syh = (sizey) / 2;
    const size_t sxh = (sizex) / 2;

    float output_sigma = std::sqrt((float) sizex * sizey) / padding * output_sigma_factor;
    float mult = -0.5 / (output_sigma * output_sigma);

    float* res_data = (float*)res.data;
    size_t index = 0;
    for (size_t i = 0; i < sizey; i++)
        for (size_t j = 0; j < sizex; j++)
        {
            const size_t ih = i - syh;
            const size_t jh = j - sxh;
            res_data[index++] = std::exp(mult * (float)(ih * ih + jh * jh));
        }
    return FFTTools::fftd(res);
}

// Obtain sub-window from image, with replication-padding and extract features
cv::Mat KCFTracker::getFeatures(const cv::Mat& image, cv::Mat& tmplate_img, const bool& inithann, int* size_patch, const float& scale_adjust)
{
    cv::Rect extracted_roi;

    const float cx = _roi.x + _roi.width / 2;
    const float cy = _roi.y + _roi.height / 2;

    if (inithann) {
        setInitialTemplateSize();
    }

    // extracted_roi为模版在实际图片中对应的区域
    extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width;
    extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height;

    // center roi with new size
    extracted_roi.x = cx - extracted_roi.width / 2;
    extracted_roi.y = cy - extracted_roi.height / 2;

    cv::Mat FeaturesMap;

    // 将图片线性插值写入_tmplate_img
    float scale = scale_adjust * _scale;
    RectTools::getTemplate(image, tmplate_img, extracted_roi);

    //// z为模版图片
    //cv::Mat z = RectTools::subwindow(image, extracted_roi, cv::BORDER_REPLICATE);

    //// 将z缩放为模版大小
    //if (z.cols != _tmpl_sz.width || z.rows != _tmpl_sz.height) {
    //    cv::resize(z, z, _tmpl_sz);
    //}

    // HOG features
    getHogFeatures(tmplate_img, FeaturesMap, size_patch); // 计算hog特征

    // Lab features
    if (_labfeatures) {
        getLabFeatures(tmplate_img, FeaturesMap, size_patch[0] * size_patch[1], size_patch);
    }

    if (inithann) {
        createHanningMats(size_patch);    // 考虑可以预先生成_hann
    }
    //FeaturesMap = _hann.mul(FeaturesMap);
    mul(FeaturesMap, _hann);
    return FeaturesMap;
}

void KCFTracker::getHogFeatures(const cv::Mat& z, cv::Mat& featureMap, int* size_patch){
    const IplImage z_ipl = cvIplImage(z);
    CvLSVMFeatureMapCaskade *map;
    getFeatureMaps(&z_ipl, cell_size, &map);
    normalizeAndTruncate(map,0.2f);
    PCAFeatureMaps(map);

    size_patch[0] = map->sizeY;
    size_patch[1] = map->sizeX;
    size_patch[2] = map->numFeatures;

    featureMap = cv::Mat(cv::Size(map->numFeatures,map->sizeX*map->sizeY), CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug
    featureMap = featureMap.t();
    freeFeatureMapObject(&map);
}

void KCFTracker::getLabFeatures(const cv::Mat& z, cv::Mat& featureMap, int ele_num, int* size_patch){
    cv::Mat imgLab;
    cvtColor(z, imgLab, CV_BGR2Lab);
    unsigned char *input = (unsigned char*)(imgLab.data);

    // Sparse output vector
    cv::Mat outputLab = cv::Mat(_labCentroids.rows, ele_num, CV_32F, float(0));

    float* outputLab_data = (float*)outputLab.data;
    int cntCell = 0;
    // Iterate through each cell
    for (int cY = cell_size; cY < z.rows-cell_size; cY+=cell_size){
        for (int cX = cell_size; cX < z.cols-cell_size; cX+=cell_size){
            // Iterate through each pixel of cell (cX,cY)
            for(int y = cY; y < cY+cell_size; ++y){
                for(int x = cX; x < cX+cell_size; ++x){
                    // Lab components for each pixel
                    float l = (float)input[(z.cols * y + x) * 3];
                    float a = (float)input[(z.cols * y + x) * 3 + 1];
                    float b = (float)input[(z.cols * y + x) * 3 + 2];

                    // Iterate trough each centroid
                    float minDist = FLT_MAX;
                    int minIdx = 0;
                    float *inputCentroid = (float*)(_labCentroids.data);
                    for(int k = 0; k < _labCentroids.rows; ++k){
                        float dist = ( (l - inputCentroid[3*k]) * (l - inputCentroid[3*k]) )
                                   + ( (a - inputCentroid[3*k+1]) * (a - inputCentroid[3*k+1]) )
                                   + ( (b - inputCentroid[3*k+2]) * (b - inputCentroid[3*k+2]) );
                        if(dist < minDist){
                            minDist = dist;
                            minIdx = k;
                        }
                    }
                    // Store result at output
                    outputLab_data[minIdx * ele_num + cntCell] += 1.0f / cell_sizeQ;
                }
            }
            cntCell++;
        }
    }
    // Update _size_patch[2] and add features to featureMap
    size_patch[2] += _labCentroids.rows;
    featureMap.push_back(outputLab);
}

void KCFTracker::setInitialTemplateSize(){
    const int padded_w = _roi.width * padding;
    const int padded_h = _roi.height * padding;

    // template_size = 96 padded = {192, 96} 则 _scale = 2 _tmpl_sz = {96, 48}
    // Fit largest dimension to the given template size
    if (padded_w >= padded_h)  //fit to width
        _scale = padded_w / (float)template_size;
    else
        _scale = padded_h / (float)template_size;

    _tmpl_sz.width = padded_w / _scale;
    _tmpl_sz.height = padded_h / _scale;

    // _tmpl_sz = {96, 48} cell_size = 4 则 _tmpl_sz = {104, 56}

    // Round to cell size and also make it even
    // 让_tmpl_sz为奇数个2*cell
    _tmpl_sz.width = (((int)(_tmpl_sz.width / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
    _tmpl_sz.height = (((int)(_tmpl_sz.height / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;

    _tmplate_img[0].create(_tmpl_sz, CV_8UC3);
    _tmplate_img[1].create(_tmpl_sz, CV_8UC3);
    _tmplate_img[2].create(_tmpl_sz, CV_8UC3);
}

// Initialize Hanning window. Function called only in the first frame.
void KCFTracker::createHanningMats(int *size_patch)
{
    cv::Mat hann1t = cv::Mat(cv::Size(size_patch[1],1), CV_32F, cv::Scalar(0));
    cv::Mat hann2t = cv::Mat(cv::Size(1, size_patch[0]), CV_32F, cv::Scalar(0));

    for (int i = 0; i < hann1t.cols; i++)
        hann1t.at<float > (0, i) = 0.5 * (1 - std::cos(2 * M_PI * i / (hann1t.cols - 1)));
    for (int i = 0; i < hann2t.rows; i++)
        hann2t.at<float > (i, 0) = 0.5 * (1 - std::cos(2 * M_PI * i / (hann2t.rows - 1)));

    const cv::Mat hann2d = hann2t * hann1t;
    // HOG features
    const cv::Mat hann1d = hann2d.reshape(1, 1); // Procedure do deal with cv::Mat multichannel bug
    _hann = cv::Mat(cv::Size(size_patch[0] * size_patch[1], size_patch[2]), CV_32F, cv::Scalar(0));
    for (int i = 0; i < size_patch[2]; i++) {
        for (int j = 0; j < size_patch[0] * size_patch[1]; j++) {
            _hann.at<float>(i, j) = hann1d.at<float>(0, j);
        }
    }
}
