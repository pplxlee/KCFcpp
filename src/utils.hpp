#include <opencv2/opencv.hpp>

/// @brief a = a.nul(b)
/// @param a 
/// @param b 
/// @return a = a * b
static cv::Mat& mul(cv::Mat& a, const cv::Mat& b)
{
    float* a_data = (float*)a.data;
    float* b_data = (float*)b.data;

    for (int i = 0; i < a.rows * a.cols; ++i)
    {
        a_data[i] *= b_data[i];
    }

    return a;
}

static float sumPow2_ch1(const cv::Mat& a)
{
    assert(a.channels() == 1);

    float* a_data = (float*)a.data;

    float res = 0.f;
    for (int i = 0; i < a.cols * a.rows; ++i)
    {
        res += a_data[i] * a_data[i];
    }

    return res;
}