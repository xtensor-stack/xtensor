
#include "xmath.hpp"
#include "xcomplex.hpp"
#include "xutils.hpp"
#include "xaxis_slice_iterator.hpp"
#include "xio.hpp"
#include "xview.hpp"

namespace xt 
{
    namespace fft
    {
        template<typename E1, typename E2, typename T = float>
        xt::xarray<std::complex<T>> fftconvolve(E1&& xvec, E2&& yvec, std::ptrdiff_t axis = -1);
        
        namespace detail
        {

            size_t reverseBits(size_t x, int n)
            {
                size_t result = 0;
                for (int i = 0; i < n; i++, x >>= 1)
                    result = (result << 1) | (x & 1U);
                return result;
            }

            bool IsPower2(size_t vecSize) noexcept
            {
                return ((vecSize & (vecSize - 1)) == 0);
            }

            template<typename E1>
            auto transformRadix2(const E1& _data)
            {

                xt::xarray<std::complex<float>> data = _data;

                // Length variables
                const size_t n(data.shape(0));
                int levels = std::floor(std::log2(n));

                // Trignometric table
                auto i = xt::linspace<float>(0, n / 2 - 1, n / 2) * std::complex<float>(0, 1);
                auto expTable = xt::eval(xt::exp(-2 * i * xt::numeric_constants<double>::PI / n));

                // Bit-reversed addressing permutation
                for (size_t i = 0; i < n; i++)
                {
                    const size_t j = reverseBits(i, levels);
                    if (j > i)
                    {
                        std::swap(data(i), data(j));
                    }
                }

                // Cooley-Tukey decimation-in-time radix-2 FFT
                // Could probably be reimplemented using stride views...
                for (size_t size = 2; size <= n; size *= 2)
                {
                    const size_t halfsize = size / 2;
                    const size_t tablestep = n / size;
                    for (size_t i = 0; i < n; i += size)
                    {
                        for (size_t j = i, k = 0; j < i + halfsize; j++, k += tablestep)
                        {
                            const auto temp = data(j + halfsize) * expTable(k);
                            data(j + halfsize) = data(j) - temp;
                            data(j) += temp;
                        }
                    }
                    // Prevent overflow in 'size *= 2'
                    if (size == n) {
                        break;
                    }
                }
                return data;
            }

            template<typename E1>
            auto transformBluestein(const E1& _data)
            {
                xt::xarray<std::complex<float>> data = _data;

                // Find a power-of-2 convolution length m such that m >= n * 2 + 1
                const size_t n(data.size());
                size_t m = 1;
                while (m / 2 <= n) 
                {
                    m *= 2;
                }

                // Trignometric table
                xt::xarray<std::complex<float>> expTable = xt::empty<std::complex<float>>({ n });
                xt::xarray<size_t> i = xt::pow(xt::linspace<size_t>(0, n - 1, n), 2);
                i %= (n * 2);
                auto angles = xt::eval(xt::numeric_constants<double>::PI * i / n);
                auto j = std::complex<float>(0, 1);
                expTable = xt::exp(-angles * j);

                // Temporary vectors and preprocessing
                xt::xarray<std::complex<float>> av = xt::empty<std::complex<float>>({ m });
                xt::view(av, xt::range(0, n)) = data * expTable;


                xt::xarray<std::complex<float>> bv = xt::empty<std::complex<float>>({ m });
                xt::view(bv, xt::range(0, n)) = xt::conj(expTable);
                xt::view(bv, xt::range(-n + 1, xt::placeholders::_)) = xt::view(xt::conj(xt::flip(expTable)), xt::range(xt::placeholders::_, - 1));

                // Convolution
                auto cv = xt::fft::fftconvolve(av, bv);

                return xt::eval(xt::view(cv, xt::range(0, n)) * expTable);
            }

            template<typename E1>
            auto fft_impl(const E1& data)
            {
                //loop through all the next highest axis

                // Is power of 2
                if (detail::IsPower2(data.size())) 
                {
                    return detail::transformRadix2(data);
                }
                //More complicated algorithm for arbitrary sizes
                else 
                {
                    return detail::transformBluestein(data);
                }
            }

            template<typename T, typename E1>
            auto fft(E1&& data, std::ptrdiff_t axis = -1)
            {
                //check the length of the data on that axis
                const size_t n(data.shape(axis));
                if (n == 0) {
                    XTENSOR_THROW(std::runtime_error, "Cannot take the FFT along an empty dimention");
                }

                //select the axis
                //create the return type
                //this can be made smarter to use floats or double for small speed up
                xt::xarray<std::complex<T>> complex_out = xt::zeros<std::complex<T>>(data.shape());
                auto saxis = xt::normalize_axis(data.dimension(), axis);
                auto begin = xt::axis_slice_begin(data, saxis);
                auto end = xt::axis_slice_end(data, saxis);
                auto iter_out = xt::axis_slice_begin(complex_out, saxis);

                for (auto iter = begin; iter != end; iter++)
                {
                    (*iter_out++) = detail::fft_impl(*iter);
                }

                return complex_out;
            }
        }

        //meta class for finding nested types inside complex types
        //false case
        template<typename T>
        struct is_complex_t : public std::false_type 
        {
        public:
            using value_type = T;
        };
        
        //true case
        template<typename T>
        struct is_complex_t<std::complex<T>> : public std::true_type 
        {
        public:
            using value_type = T;
        };

        //case for integer types... We will always opt for a single precision
        template<class E1>
        auto fft(E1 e1, std::ptrdiff_t axis = -1, typename std::enable_if<std::is_integral<typename E1::value_type>::value>::type* = nullptr)
        {
            return detail::fft<float>(e1, axis);
        }

        //Case for floating point types
        template<class E1>
        auto fft(E1 e1, std::ptrdiff_t axis = -1, typename std::enable_if<std::is_floating_point<typename E1::value_type>::value>::type* = nullptr)
        {
            return detail::fft<typename E1::value_type>(e1, axis);
        }

        //Final case for a complex type. We want to know the inner type of the complex value to match precision so we
        //use the meta class is_complex_t to find the nested type
        template<class E1>
        auto fft(E1 e1, std::ptrdiff_t axis = -1, typename std::enable_if<is_complex_t<typename E1::value_type>::value>::type* = nullptr)
        {
            return detail::fft<typename is_complex_t<typename E1::value_type>::value_type>(e1, axis);
        }

        namespace detail 
        {
            template<typename T, typename E1>
            auto ifft(E1&& data, std::ptrdiff_t axis = -1)
            {
                //check the length of the data on that axis
                const size_t n = data.shape(axis);
                if (n == 0) {
                    XTENSOR_THROW(std::runtime_error, "Cannot take the iFFT along an empty dimention");
                }
                xt::xarray<std::complex<T>> complex_args = data;
                complex_args = xt::conj(complex_args);
                auto fft_res = xt::fft::fft(complex_args, axis);
                fft_res = xt::conj(fft_res);
                return fft_res;
            }
        }

        //Case for floating point types
        template<class E1>
        auto ifft(E1 e1, std::ptrdiff_t axis = -1, typename std::enable_if<std::is_floating_point<typename E1::value_type>::value>::type* = nullptr)
        {
            return detail::ifft<typename E1::value_type>(e1, axis);
        }

        //case for integer types... We will always opt for a single precision
        template<class E1>
        auto ifft(E1 e1, std::ptrdiff_t axis = -1, typename std::enable_if<std::is_integral<typename E1::value_type>::value>::type* = nullptr)
        {
            return detail::ifft<float>(e1, axis);
        }

        //Final case for a complex type. We want to know the inner type of the complex value to match precision so we
        //use the meta class is_complex_t to find the nested type
        template<class E1>
        auto ifft(E1 e1, std::ptrdiff_t axis = -1, typename std::enable_if<is_complex_t<typename E1::value_type>::value>::type* = nullptr)
        {
            return detail::ifft<typename is_complex_t<typename E1::value_type>::value_type>(e1, axis);
        }

        template<typename E1, typename E2, typename T>
        xt::xarray<std::complex<T>> fftconvolve(E1&& xvec, E2&& yvec, std::ptrdiff_t axis)
        {
            //we could broadcast but that could get complicated???
            if (xvec.dimension() != yvec.dimension())
            {
                XTENSOR_THROW(std::runtime_error, "Mismatched dimentions");
            }

            auto saxis = xt::normalize_axis(xvec.dimension(), axis);
            if (xvec.shape(saxis) != yvec.shape(saxis))
            {
                XTENSOR_THROW(std::runtime_error, "Mismatched lengths along slice axis");
            }
   
            const size_t n = xvec.shape(saxis);

            auto xv = fft(xvec, axis);
            auto yv = fft(yvec, axis);

            auto begin_x = xt::axis_slice_begin(xv, saxis);
            auto end_x = xt::axis_slice_end(xv, saxis);
            auto iter_y = xt::axis_slice_begin(yv, saxis);

            for (auto iter = begin_x; iter != end_x; iter++) 
            {
                (*iter) = (*iter_y++)*(*iter);
            }
            
            auto outvec = ifft(xv, axis);

            // Scaling (because this FFT implementation omits it)
            outvec = outvec / n;

            return outvec;
        }
    }
}
