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

#include <opencv2/imgproc/types_c.h>
#include <math.h>

#ifndef _OPENCV_RECTTOOLS_HPP_
#define _OPENCV_RECTTOOLS_HPP_
#endif

namespace RectTools
{

template <typename t>
inline cv::Vec<t, 2 > center(const cv::Rect_<t> &rect)
{
    return cv::Vec<t, 2 > (rect.x + rect.width / (t) 2, rect.y + rect.height / (t) 2);
}

template <typename t>
inline t x2(const cv::Rect_<t> &rect)
{
    return rect.x + rect.width;
}

template <typename t>
inline t y2(const cv::Rect_<t> &rect)
{
    return rect.y + rect.height;
}

template <typename t>
inline void resize(cv::Rect_<t> &rect, float scalex, float scaley = 0)
{
    if (!scaley)scaley = scalex;
    rect.x -= rect.width * (scalex - 1.f) / 2.f;
    rect.width *= scalex;

    rect.y -= rect.height * (scaley - 1.f) / 2.f;
    rect.height *= scaley;

}

template <typename t>
inline void limit(cv::Rect_<t> &rect, cv::Rect_<t> limit)
{
    if (rect.x + rect.width > limit.x + limit.width)rect.width = (limit.x + limit.width - rect.x);
    if (rect.y + rect.height > limit.y + limit.height)rect.height = (limit.y + limit.height - rect.y);
    if (rect.x < limit.x)
    {
        rect.width -= (limit.x - rect.x);
        rect.x = limit.x;
    }
    if (rect.y < limit.y)
    {
        rect.height -= (limit.y - rect.y);
        rect.y = limit.y;
    }
    if(rect.width<0)rect.width=0;
    if(rect.height<0)rect.height=0;
}

template <typename t>
inline void limit(cv::Rect_<t> &rect, t width, t height, t x = 0, t y = 0)
{
    limit(rect, cv::Rect_<t > (x, y, width, height));
}

template <typename t>
inline cv::Rect getBorder(const cv::Rect_<t > &original, cv::Rect_<t > & limited)
{
    cv::Rect_<t > res;
    res.x = limited.x - original.x;
    res.y = limited.y - original.y;
    res.width = x2(original) - x2(limited);
    res.height = y2(original) - y2(limited);
    assert(res.x >= 0 && res.y >= 0 && res.width >= 0 && res.height >= 0);
    return res;
}

inline cv::Mat subwindow(const cv::Mat &in, const cv::Rect & window, int borderType = cv::BORDER_CONSTANT)
{
    cv::Rect cutWindow = window;
    RectTools::limit(cutWindow, in.cols, in.rows);
    if (cutWindow.height <= 0 || cutWindow.width <= 0)assert(0); //return cv::Mat(window.height,window.width,in.type(),0) ;
    cv::Rect border = RectTools::getBorder(window, cutWindow);
    cv::Mat res = in(cutWindow);

    if (border != cv::Rect(0, 0, 0, 0))
    {
        cv::copyMakeBorder(res, res, border.y, border.height, border.x, border.width, borderType);
    }
    return res;
}

inline cv::Mat getGrayImage(cv::Mat img)
{
    cv::cvtColor(img, img, CV_BGR2GRAY);
    img.convertTo(img, CV_32F, 1 / 255.f);
    return img;
}

// template_img should be pre-allocated memory
// scale the img's roi to template_img
// bilinear
inline void scaleRoiImg(const cv::Mat& img, cv::Mat& template_img, const cv::Rect roi)
{
    constexpr static int FIX_POINT_BASE = 10;

    assert(!template_img.empty());
    assert(img.channels() == 3 && template_img.channels() == 3);

    int scale_w = (roi.width << FIX_POINT_BASE) / template_img.cols;
    int scale_h = (roi.height << FIX_POINT_BASE) / template_img.rows;
    uchar* iptr = img.data;
    uchar* tptr = template_img.data;
    int map_x, map_y;
    int mx[2], my[2];
    int index[4];
    int w[4];
    for (int y = 0; y < template_img.rows; ++y)
    {
        for (int x = 0; x < template_img.cols; ++x)
        {
            map_x = (roi.x << FIX_POINT_BASE) + x * scale_w;
            map_y = (roi.y << FIX_POINT_BASE) + y * scale_h;

            if (map_x < 0)
            {
                mx[0] = mx[1] = 0;
                w[0] = 1 << FIX_POINT_BASE;
                w[1] = 0;
            }
            else if (map_x >= ((img.cols - 1) << FIX_POINT_BASE))
            {
                mx[0] = mx[1] = img.cols - 1;
                w[0] = 1 << FIX_POINT_BASE;
                w[1] = 0;
            }
            else
            {
                mx[0] = map_x >> FIX_POINT_BASE;
                mx[1] = mx[0] + 1;
                w[0] = (mx[1] << FIX_POINT_BASE) - map_x;
                w[1] = (1 << FIX_POINT_BASE) - w[0];
            }

            if (map_y < 0)
            {
                my[0] = my[1] = 0;
                w[2] = 1 << FIX_POINT_BASE;
                w[3] = 0;
            }
            else if (map_y >= ((img.rows - 1) << FIX_POINT_BASE))
            {
                my[0] = my[1] = img.rows - 1;
                w[2] = 1 << FIX_POINT_BASE;
                w[3] = 0;
            }
            else
            {
                my[0] = map_y >> FIX_POINT_BASE;
                my[1] = my[0] + 1;
                w[2] = (my[1] << FIX_POINT_BASE) - map_y;
                w[3] = (1 << FIX_POINT_BASE) - w[2];
            }

            // 0 1
            // 2 3
            index[0] = (my[0] * img.cols + mx[0]) * 3;
            index[1] = (my[0] * img.cols + mx[1]) * 3;
            index[2] = (my[1] * img.cols + mx[0]) * 3;
            index[3] = (my[1] * img.cols + mx[1]) * 3;

            tptr[0] = ((iptr[index[0]] * w[0] + iptr[index[1]] * w[1]) * w[2] +
                (iptr[index[2]] * w[0] + iptr[index[3]] * w[1]) * w[3]) >> (2 * FIX_POINT_BASE);
            tptr[1] = ((iptr[index[0] + 1] * w[0] + iptr[index[1] + 1] * w[1]) * w[2] +
                (iptr[index[2] + 1] * w[0] + iptr[index[3] + 1] * w[1]) * w[3]) >> (2 * FIX_POINT_BASE);
            tptr[2] = ((iptr[index[0] + 2] * w[0] + iptr[index[1] + 2] * w[1]) * w[2] +
                (iptr[index[2] + 2] * w[0] + iptr[index[3] + 2] * w[1]) * w[3]) >> (2 * FIX_POINT_BASE);

            tptr += 3;
        }
    }
}

}



