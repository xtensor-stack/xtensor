#ifndef XFFT_HPP
#define XFFT_HPP

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
        template<
            typename T = double, 
            typename E1, 
            typename E2
        >
        xt::xarray<std::complex<T>> fft_convolve(E1&& xvec, E2&& yvec, std::ptrdiff_t axis = -1);
        
        namespace detail
        {
            template<typename T>
            T pi()
            {
                constexpr double PI = 3.141592653589793238463;
                return PI;
            }

            std::size_t reverse_bits(std::size_t x, int n)
            {
                std::size_t result = 0;
                for (int i = 0; i < n; i++, x >>= 1)
                    result = (result << 1) | (x & 1U);
                return result;
            }

            template<typename E1>
            bool is_power2(const E1& data) noexcept
            {
                return ((data.size() & (data.size() - 1)) == 0);
            }

            template<
                typename T
            >
            auto transform_radix2(xt::xarray<std::complex<T>> data)
            {
                // Length variables
                const std::size_t n = data.shape(0);
                int levels = std::floor(std::log2(n));

                // Trignometric table
                auto i = xt::linspace<T>(0, n / 2 - 1, n / 2) * std::complex<T>(0, 1);
                //I would like to do this but there appears to be a compiler error in CPP14
                //Works in CPP17 and MSVC CPP14
                //Linker error when using numeric_constants in this header
                //auto expTable = xt::eval(xt::exp(-2 * i * ::xt::numeric_constants<T>().PI / n));
                auto expTable = xt::eval(xt::exp(-2 * i * pi<T>() / n));

                // Bit-reversed addressing permutation
                for (std::size_t i = 0; i < n; i++)
                {
                    const std::size_t j = reverse_bits(i, levels);
                    if (j > i)
                    {
                        std::swap(data(i), data(j));
                    }
                }

                // Cooley-Tukey decimation-in-time radix-2 FFT
                // Could probably be reimplemented using stride views...
                for (std::size_t size = 2; size <= n; size *= 2)
                {
                    const std::size_t halfsize = size / 2;
                    const std::size_t tablestep = n / size;
                    for (std::size_t i = 0; i < n; i += size)
                    {
                        for (std::size_t j = i, k = 0; j < i + halfsize; j++, k += tablestep)
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

            template<
                typename T
                >
            auto transform_bluestein(xt::xarray<std::complex<T>> data)
            {
                // Find a power-of-2 convolution length m such that m >= n * 2 + 1
                const std::size_t n = data.size();
                std::size_t m = 1;
                while (m / 2 <= n) 
                {
                    m *= 2;
                }

                // Trignometric table
                auto expTable = xt::xtensor<std::complex<T>, 1>::from_shape({ n });
                xt::xtensor<std::size_t, 1> i = xt::pow(xt::linspace<std::size_t>(0, n - 1, n), 2);
                i %= (n * 2);
                
                //I would like to do this but there appears to be a compiler error in CPP14
                //Works in CPP17 and MSVC CPP14
                //Linker error when using numeric_constants in this header
                //auto angles = xt::eval(::xt::numeric_constants<T>::PI * i / n);
                auto angles = xt::eval(pi<T>() * i / n);
                auto j = std::complex<T>(0, 1);
                expTable = xt::exp(-angles * j);

                // Temporary vectors and preprocessing
                xt::xarray<std::complex<T>> av = xt::empty<std::complex<T>>({ m });
                xt::view(av, xt::range(0, n)) = data * expTable;


                xt::xarray<std::complex<T>> bv = xt::empty<std::complex<T>>({ m });
                xt::view(bv, xt::range(0, n)) = xt::conj(expTable);
                xt::view(bv, xt::range(-n + 1, xt::placeholders::_)) = xt::view(xt::conj(xt::flip(expTable)), xt::range(xt::placeholders::_, - 1));

                // Convolution
                auto cv = xt::fft::fft_convolve<T>(av, bv);

                return xt::eval(xt::view(cv, xt::range(0, n)) * expTable);
            }

            template<
                typename T
            >
            auto fft_impl(xt::xarray<std::complex<T>> data)
            {
                //loop through all the next highest axis

                // Is power of 2
                if (detail::is_power2(data))
                {
                    return detail::transform_radix2<T>(std::move(data));
                }
                //More complicated algorithm for arbitrary sizes
                else 
                {
                    return detail::transform_bluestein<T>(std::move(data));
                }
            }

            template<
                typename T, 
                typename E1
            >
            auto fft(E1&& data, std::ptrdiff_t axis = -1)
            {
                //check the length of the data on that axis
                const std::size_t n(data.shape(axis));
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
                    (*iter_out++) = detail::fft_impl<T>(*iter);
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
            template<
                typename T, 
                typename E1
            >
            auto ifft(E1&& data, std::ptrdiff_t axis = -1)
            {
                //check the length of the data on that axis
                const std::size_t n = data.shape(axis);
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

        /*
        * @brief performs a circular fft convolution xvec and yvec must
        *        be the same shape.
        * @param xvec first array of the convolution
        * @param yvec second array of the convolution
        * @param axis axis along which to perform the convolution
        */
        template<
            typename T, 
            typename E1, 
            typename E2
        >
        xt::xarray<std::complex<T>> fft_convolve(E1&& xvec, E2&& yvec, std::ptrdiff_t axis)
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
   
            const std::size_t n = xvec.shape(saxis);

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

#endif