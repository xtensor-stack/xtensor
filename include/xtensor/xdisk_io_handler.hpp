#ifndef XTENSOR_DISK_IO_HANDLER_HPP
#define XTENSOR_DISK_IO_HANDLER_HPP

#include "xarray.hpp"
#include "xexpression.hpp"

namespace xt
{
    template <class F>
    class xdisk_io_handler
    {
    public:

        template <class E>
        void write(xexpression<E>& e, std::string& path)
        {
            std::ofstream m_out_file(path, std::ofstream::binary);
            if (m_out_file.is_open())
            {
                m_format.dump(m_out_file, e);
            }
            else
            {
                std::runtime_error("write: failed to open file " + path);
            }
        }

        template <class EC>
        void read(xarray<EC>& a, std::string& path)
        {
            // not all formats store the shape (e.g. Blosc)
            // so we reshape after loading
            std::ifstream m_in_file(path, std::ifstream::binary);
            const auto shape = a.shape();
            if (m_in_file.is_open())
            {
                m_format.load(m_in_file, a);
                a.reshape(shape);
            }
            else
            {
                a = broadcast(0, shape);
            }
        }

        template <class C>
        void configure(C& config)
        {
            m_format.configure(config);
        }

    private:
        F m_format;
    };
}

#endif
