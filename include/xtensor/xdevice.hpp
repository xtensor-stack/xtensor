#ifndef XTENSOR_DEVICE_HPP
#define XTENSOR_DEVICE_HPP

#include <memory>
#include <algorithm>
#include <functional>
#include <vector>

namespace xt{
    namespace detail{

    }
    /**
     * Device implementation for the various operations. All device specific code goes in here disabled via macro
     * for invalid syntax which might be needed for Sycl or CUDA.
     */
//#ifdef XTENSOR_DEVICE_ASSIGN
    template<class T>
    class host_device_batch
    {
    public:
        host_device_batch(const T* ptr, std::size_t size)
        {
            //copy the data to the device
            //CUDA Impl = Nearly identical
            m_data.resize(size);
            std::copy(ptr, ptr + size, std::begin(m_data));
        }
        template<class A>
        host_device_batch& operator+(const host_device_batch<A>& rhs)
        {
            //CUDA impl = thrust::transform(m_data.begin(), m_data.end(), rhs.m_data().begin(), m_data.end(), thrust::plus<T>{});
            std::transform(std::begin(m_data), std::end(m_data), std::begin(rhs.m_data), std::begin(m_data), std::plus<T>{});
            return *this;
        }
        template<class A>
        host_device_batch& operator-(const host_device_batch<A>& rhs)
        {
            std::transform(std::begin(m_data), std::end(m_data), std::begin(rhs.m_data), std::begin(m_data), std::minus<T>{});
            return *this;
        }
        template<class A>
        host_device_batch& operator*(const host_device_batch<A>& rhs)
        {
            std::transform(std::begin(m_data), std::end(m_data), std::begin(rhs.m_data), std::begin(m_data), std::multiplies<T>{});
            return *this;
        }
        template<class A>
        host_device_batch& operator/(const host_device_batch<A>& rhs)
        {
            std::transform(std::begin(m_data), std::end(m_data), std::begin(rhs.m_data), std::begin(m_data), std::divides<T>{});
            return *this;
        }
        void store_host(T* dst)
        {
            std::copy(std::begin(m_data), std::end(m_data), dst);
        }
    private:
        //CUDA impl = thrust::device_vector<T> m_data;
        std::vector<T> m_data;
    };
//#endif

    // template<class T>
    // class cuda_device_batch : public batch<host_device_batch<T>>
    // {
    // public:
        
    // };

    // template<class T>
    // class intel_device_batch : public batch<host_device_batch<T>>
    // {
    // public:
        
    // };

    // template<class T>
    // class opencl_device_batch : public batch<host_device_batch<T>>
    // {
    // public:
        
    // };
}

#endif
