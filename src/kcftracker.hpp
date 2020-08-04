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

#pragma once

#include "tracker.h"

#ifndef _OPENCV_KCFTRACKER_HPP_
#define _OPENCV_KCFTRACKER_HPP_
#endif

class KCFTracker : public Tracker
{
public:
    // Constructor
    KCFTracker(const bool& lab = true, const bool& multi_thread = true);
    ~KCFTracker();

    // Initialize tracker 
    virtual void init(const cv::Rect &roi, const cv::Mat& image);
    
    // Update position based on the new frame
    virtual cv::Rect update(const cv::Mat& image, float& prob);

    float interp_factor; // linear interpolation factor for adaptation
    float interp_threshold; // the threshold for linear interpolation
    float sigma; // gaussian kernel bandwidth
    float lambda; // regularization
    int cell_size; // HOG cell size
    int cell_sizeQ; // cell size^2, to avoid repeated operations
    float padding; // extra area surrounding the target
    float output_sigma_factor; // bandwidth of gaussian target
    int template_size; // template size
    float scale_step; // scale step for multi-scale estimation
    float scale_weight;  // to downweight detection scores of other scales for added stability

protected:
    // Set default parameters
    void setParameters();

    // Detect object in the current frame.
    cv::Point2f detect(const cv::Mat& z, const cv::Mat& x, float &peak_value, int* size_patch);

    // train tracker with a single image
    void train(const cv::Mat& x, const float& train_interp_factor, int* size_patch);

    // Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
    cv::Mat gaussianCorrelation(const cv::Mat& x1, const cv::Mat& x2, int* size_patch);
    cv::Mat gaussianCorrelation(const cv::Mat& x, int* size_patch);

    // Create Gaussian Peak. Function called only in the first frame.
    cv::Mat createGaussianPeak(const size_t& sizey, const size_t& sizex);

    // Obtain sub-window from image, with replication-padding and extract features
    cv::Mat getFeatures(const cv::Mat& image, const bool& inithann, int* size_patch, const float& scale_adjust = 1.0f);

    // Get hog features
    void getHogFeatures(const cv::Mat& z, cv::Mat& featureMap, int *size_patch);

    // Get lab features
    void getLabFeatures(const cv::Mat& z, cv::Mat& featureMap, int ele_num, int *size_patch);

    // Set initial template size, only runs in the first frame
    void setInitialTemplateSize();

    // Initialize Hanning window. Function called only in the first frame.
    void createHanningMats(int *size_patch);

    // Calculate sub-pixel peak for one dimension
    float subPixelPeak(const float& left, const float& center, const float& right)
    {
        const float divisor = 2 * center - right - left;
        return ((divisor == 0) ? 0 : (0.5f * (right - left) / divisor));
    }

    cv::Mat _alphaf;
    cv::Mat _prob;
    cv::Mat _tmpl;
    cv::Mat _num;
    cv::Mat _den;
    cv::Mat _labCentroids;

private:
    int _size_patch[3];
    cv::Mat _hann;
    cv::Size _tmpl_sz;
    float _scale;
    int _gaussian_size;
    //bool _hogfeatures;
    bool _labfeatures;
    //bool _multiscale;
    //bool _fixed_window;

    class MultiThreadHelper;
    MultiThreadHelper *multi_thread_helper_;
};
