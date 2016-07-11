#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <vector>
#include <algorithm>
#include <iterator>
#include <iostream>

namespace ndarray
{
    template <class S>
    void print_vector(const std::vector<S>& data)
    {
        std::copy(data.begin(), data.end(), std::ostream_iterator<S>(std::cout, ", "));
    }
}

#endif

