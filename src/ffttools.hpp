/* 
Author: Christian Bailer
Contact address: Christian.Bailer@dfki.de 
Department Augmented Vision DFKI 

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

//#include <cv.h>

#ifndef _OPENCV_FFTTOOLS_HPP_
#define _OPENCV_FFTTOOLS_HPP_
#endif

//NOTE: FFTW support is still shaky, disabled for now.
/*#ifdef USE_FFTW
#include <fftw3.h>
#endif*/

namespace FFTTools
{
// Previous declarations, to avoid warnings
cv::Mat fftd(cv::Mat img, bool backwards = false);
cv::Mat real(cv::Mat img);
cv::Mat imag(cv::Mat img);
cv::Mat magnitude(cv::Mat img);
cv::Mat complexMultiplication(cv::Mat a, cv::Mat b);
cv::Mat complexDivision(cv::Mat a, cv::Mat b);
void rearrange(cv::Mat &img);
void normalizedLogTransform(cv::Mat &img);


cv::Mat fftd(cv::Mat img, bool backwards)
{
/*
#ifdef USE_FFTW

    fftw_complex * fm = (fftw_complex*) fftw_malloc(sizeof (fftw_complex) * img.cols * img.rows);

    fftw_plan p = fftw_plan_dft_2d(img.rows, img.cols, fm, fm, backwards ? 1 : -1, 0 * FFTW_ESTIMATE);


    if (img.channels() == 1)
    {
        for (int i = 0; i < img.rows; i++)
            for (int j = 0; j < img.cols; j++)
            {
                fm[i * img.cols + j][0] = img.at<float>(i, j);
                fm[i * img.cols + j][1] = 0;
            }
    }
    else
    {
        assert(img.channels() == 2);
        for (int i = 0; i < img.rows; i++)
            for (int j = 0; j < img.cols; j++)
            {
                fm[i * img.cols + j][0] = img.at<cv::Vec2d > (i, j)[0];
                fm[i * img.cols + j][1] = img.at<cv::Vec2d > (i, j)[1];
            }
    }
    fftw_execute(p);
    cv::Mat res(img.rows, img.cols, CV_64FC2);


    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
        {
            res.at<cv::Vec2d > (i, j)[0] = fm[i * img.cols + j][0];
            res.at<cv::Vec2d > (i, j)[1] = fm[i * img.cols + j][1];

            //  _iout(fm[i * img.cols + j][0]);
        }

    if (backwards)res *= 1.d / (float) (res.cols * res.rows);

    fftw_free(p);
    fftw_free(fm);
    return res;

#else
*/
    if (img.channels() == 1)
    {
        cv::Mat planes[] = {cv::Mat_<float> (img), cv::Mat_<float>::zeros(img.size())};
        //cv::Mat planes[] = {cv::Mat_<double> (img), cv::Mat_<double>::zeros(img.size())};
        cv::merge(planes, 2, img);
    }
    cv::dft(img, img, backwards ? (cv::DFT_INVERSE | cv::DFT_SCALE) : 0 );

    return img;

/*#endif*/

}

cv::Mat real(cv::Mat img)
{
    std::vector<cv::Mat> planes;
    cv::split(img, planes);
    return planes[0];
}

cv::Mat imag(cv::Mat img)
{
    std::vector<cv::Mat> planes;
    cv::split(img, planes);
    return planes[1];
}

/// @brief a = a + real(b)
/// @param a 
/// @param b 
/// @return a
cv::Mat& addReal_float(cv::Mat& a, const cv::Mat& b)
{
    float* a_data = (float*)a.data;
    float* b_data = (float*)b.data;

    int ch = b.channels();

    for (int i = 0; i < a.rows * a.cols; ++i)
    {
        a_data[i] += b_data[i * ch];
    }

    return a;
}

/// @brief rearrange(b); a = a + real(b)
/// @param a 
/// @param b 
/// @return a
cv::Mat& addRearrangeReal_float(cv::Mat& a, const cv::Mat& b)
{
    float* a_data = (float*)a.data;
    float* b_data = (float*)b.data;

    int ch = b.channels();

    int cx = a.cols / 2;
    int cy = a.rows / 2;

    int delta_tl_br = cy * a.cols + cx;
    int delta_tr_bl = cy * a.cols - cx;
    int dalta[4] = { delta_tl_br , delta_tr_bl , -delta_tr_bl , -delta_tl_br };

    int i = 0;
    for (int y = 0; y < a.rows; ++y)
    {
        for (int x = 0; x < a.cols; ++x)
        {
            a_data[i] += b_data[(i + dalta[((y >= cy) << 1) | (x >= cx)]) * ch];
            ++i;
        }
    }

    return a;
}

cv::Mat magnitude(cv::Mat img)
{
    cv::Mat res;
    std::vector<cv::Mat> planes;
    cv::split(img, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    if (planes.size() == 1) res = cv::abs(img);
    else if (planes.size() == 2) cv::magnitude(planes[0], planes[1], res); // planes[0] = magnitude
    else assert(0);
    return res;
}

cv::Mat complexMultiplication(cv::Mat a, cv::Mat b)
{
    cv::Mat res(cv::Size(a.cols, a.rows), CV_32FC2);
    float* a_data = (float*)a.data;
    float* b_data = (float*)b.data;
    float* res_data = (float*)res.data;

    int a_ch = a.channels();
    int b_ch = b.channels();

    int num = a.cols * a.rows;
    for (int i = 0; i < num; ++i)
    {
        auto i_a = i * a_ch;
        auto i_b = i * b_ch;
        res_data[i * 2] = a_data[i_a] * b_data[i_b] - a_data[i_a + 1] * b_data[i_b + 1];
        res_data[i * 2 + 1] = a_data[i_a] * b_data[i_b + 1] - a_data[i_a + 1] * b_data[i_b];
    }

    return res;
}

cv::Mat complexDivision(cv::Mat a, cv::Mat b)
{
    cv::Mat res(cv::Size(a.cols, a.rows), CV_32FC2);
    float* a_data = (float*)a.data;
    float* b_data = (float*)b.data;
    float* res_data = (float*)res.data;

    int a_ch = a.channels();
    int b_ch = b.channels();

    int num = a.cols * a.rows;
    for (int i = 0; i < num; ++i)
    {
        auto i_a = i * a_ch;
        auto i_b = i * b_ch;
        float div = 1.f / (b_data[i_b] * b_data[i_b] + b_data[i_b + 1] * b_data[i_b + 1]);
        res_data[i * 2] = (a_data[i_a] * b_data[i_b] - a_data[i_a + 1] * b_data[i_b + 1]) * div;
        res_data[i * 2 + 1] = (a_data[i_a] * b_data[i_b + 1] - a_data[i_a + 1] * b_data[i_b]) * div;
    }
    return res;
}

void rearrange(cv::Mat &img)
{
    // img = img(cv::Rect(0, 0, img.cols & -2, img.rows & -2));
    int cx = img.cols / 2;
    int cy = img.rows / 2;

    cv::Mat q0(img, cv::Rect(0, 0, cx, cy)); // Top-Left - Create a ROI per quadrant
    cv::Mat q1(img, cv::Rect(cx, 0, cx, cy)); // Top-Right
    cv::Mat q2(img, cv::Rect(0, cy, cx, cy)); // Bottom-Left
    cv::Mat q3(img, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp; // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}
/*
template < typename type>
cv::Mat fouriertransFull(const cv::Mat & in)
{
    return fftd(in);

    cv::Mat planes[] = {cv::Mat_<type > (in), cv::Mat_<type>::zeros(in.size())};
    cv::Mat t;
    assert(planes[0].depth() == planes[1].depth());
    assert(planes[0].size == planes[1].size);
    cv::merge(planes, 2, t);
    cv::dft(t, t);

    //cv::normalize(a, a, 0, 1, CV_MINMAX);
    //cv::normalize(t, t, 0, 1, CV_MINMAX);

    // cv::imshow("a",real(a));
    //  cv::imshow("b",real(t));
    // cv::waitKey(0);

    return t;
}*/

void normalizedLogTransform(cv::Mat &img)
{
    img = cv::abs(img);
    img += cv::Scalar::all(1);
    cv::log(img, img);
    // cv::normalize(img, img, 0, 1, CV_MINMAX);
}

}
