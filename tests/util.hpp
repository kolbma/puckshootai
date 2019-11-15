#ifndef PUCKETSHOOTAI_UTIL_HPP
#define PUCKETSHOOTAI_UTIL_HPP

#include <cmath>
#include <cstdint>

template <typename T>
bool is_almost_equal(const T a, const T b, const uint8_t decimal_accuracy)
{
    const T ac = std::round(a * (10 ^ decimal_accuracy)) / (10 ^ decimal_accuracy);
    const T bc = std::round(b * (10 ^ decimal_accuracy)) / (10 ^ decimal_accuracy);

    return ac == bc;
}

#endif // PUCKETSHOOTAI_UTIL_HPP
