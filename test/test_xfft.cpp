#include "xtensor/xarray.hpp"
#include "xtensor/xfft.hpp"

#include "test_common_macros.hpp"

namespace xt
{
    TEST(xfft, fft_power_2)
    {
        size_t k = 2;
        size_t n = 8192;
        size_t A = 10;
        auto x = xt::linspace<float>(0, static_cast<float>(n - 1), n);
        xt::xarray<float> y = A * xt::sin(2 * xt::numeric_constants<float>::PI * x * k / n);
        auto res = xt::fft::fft(y) / (n / 2);
        REQUIRE(A == doctest::Approx(std::abs(res(k))).epsilon(.0001));
    }

    TEST(xfft, ifft_power_2)
    {
        size_t k = 2;
        size_t n = 8;
        size_t A = 10;
        auto x = xt::linspace<float>(0, static_cast<float>(n - 1), n);
        xt::xarray<float> y = A * xt::sin(2 * xt::numeric_constants<float>::PI * x * k / n);
        auto res = xt::fft::ifft(y) / (n / 2);
        REQUIRE(A == doctest::Approx(std::abs(res(k))).epsilon(.0001));
    }

    TEST(xfft, convolve_power_2)
    {
        xt::xarray<float> x = {1.0, 1.0, 1.0, 5.0};
        xt::xarray<float> y = {5.0, 1.0, 1.0, 1.0};
        xt::xarray<float> expected = {12, 12, 12, 28};

        auto result = xt::fft::convolve(x, y);

        for (size_t i = 0; i < x.size(); i++)
        {
            REQUIRE(expected(i) == doctest::Approx(std::abs(result(i))).epsilon(.0001));
        }
    }

    TEST(xfft, fft_n_0_axis)
    {
        size_t k = 2;
        size_t n = 10;
        size_t A = 1;
        size_t dim = 10;
        auto x = xt::linspace<float>(0, n - 1, n) * xt::ones<float>({dim, n});
        xt::xarray<float> y = A * xt::sin(2 * xt::numeric_constants<float>::PI * x * k / n);
        y = xt::transpose(y);
        auto res = xt::fft::fft(y, 0) / (n / 2.0);
        REQUIRE(A == doctest::Approx(std::abs(res(k, 0))).epsilon(.0001));
        REQUIRE(A == doctest::Approx(std::abs(res(k, 1))).epsilon(.0001));
    }

    TEST(xfft, fft_n_1_axis)
    {
        size_t k = 2;
        size_t n = 15;
        size_t A = 1;
        size_t dim = 2;
        auto x = xt::linspace<float>(0, n - 1, n) * xt::ones<float>({dim, n});
        xt::xarray<float> y = A * xt::sin(2 * xt::numeric_constants<float>::PI * x * k / n);
        auto res = xt::fft::fft(y) / (n / 2.0);
        REQUIRE(A == doctest::Approx(std::abs(res(0, k))).epsilon(.0001));
        REQUIRE(A == doctest::Approx(std::abs(res(1, k))).epsilon(.0001));
    }

    TEST(xfft, convolve_n)
    {
        xt::xarray<float> x = {1.0, 1.0, 1.0, 5.0, 1.0};
        xt::xarray<float> y = {5.0, 1.0, 1.0, 1.0, 1.0};
        xt::xarray<size_t> expected = {13, 13, 13, 29, 13};

        auto result = xt::fft::convolve(x, y);

        xt::xarray<float> abs = xt::abs(result);

        for (size_t i = 0; i < abs.size(); i++)
        {
            REQUIRE(expected(i) == doctest::Approx(abs(i)).epsilon(.0001));
        }
    }
}
