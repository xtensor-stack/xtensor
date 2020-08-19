#ifndef XTENSOR_DISK_IO_HANDLER_HPP
#define XTENSOR_DISK_IO_HANDLER_HPP

#include "xarray.hpp"
#include "xexpression.hpp"

namespace xt
{
    template <class C>
    class xdisk_io_handler
    {
    public:
        template <class E>
        void write(xexpression<E>& expression, std::string& path)
        {
            std::ofstream m_out_file(path, std::ofstream::binary);
            if (m_out_file.is_open())
            {
                dump_file(m_out_file, expression, m_format_config);
            }
            else
            {
                std::runtime_error("write: failed to open file " + path);
            }
        }

        template <class ET>
        void read(ET& array, std::string& path)
        {
            // not all formats store the shape (e.g. Blosc)
            // so we reshape after loading
            std::ifstream m_in_file(path, std::ifstream::binary);
            const auto shape = array.shape();
            if (m_in_file.is_open())
            {
                array = load_file<ET>(m_in_file, m_format_config);
                array.reshape(shape);
            }
            else
            {
                array = zeros<typename ET::value_type>(shape);
            }
        }

        void configure_format(C& format_config)
        {
            m_format_config = format_config;
        }

    private:
        C m_format_config;
    };
}

#endif
