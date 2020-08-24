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
        void write(const xexpression<E>& expression, const std::string& path) const;

        template <class ET>
        void read(ET& array, const std::string& path, bool throw_on_fail = false) const;

        void configure_format(const C& format_config);

    private:

        C m_format_config;
    };

    template <class C>
    template <class E>
    inline void xdisk_io_handler<C>::write(const xexpression<E>& expression, const std::string& path) const
    {
        std::ofstream out_file(path, std::ofstream::binary);
        if (out_file.is_open())
        {
            dump_file(out_file, expression, m_format_config);
        }
        else
        {
            std::runtime_error("write: failed to open file " + path);
        }
    }

    template <class C>
    template <class ET>
    inline void xdisk_io_handler<C>::read(ET& array, const std::string& path, bool throw_on_fail) const
    {
        std::ifstream in_file(path, std::ifstream::binary);
        if (in_file.is_open())
        {
            load_file<ET>(in_file, array, m_format_config);
        }
        else
        {
            if (throw_on_fail)
            {
                XTENSOR_THROW(std::runtime_error, "read: failed to open file " + path);
            }
            else
            {
                auto shape = array.shape();
                array = zeros<typename ET::value_type>(shape);
            }
        }
    }

    template <class C>
    inline void xdisk_io_handler<C>::configure_format(const C& format_config)
    {
        m_format_config = format_config;
    }


}

#endif
