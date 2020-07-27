#ifndef XTENSOR_DISK_IO_HANDLER_HPP
#define XTENSOR_DISK_IO_HANDLER_HPP

#include "xarray.hpp"
#include "xcsv.hpp"

namespace xt
{
    template <class EC>
    class xdisk_io_handler
    {
    public:

        void write(xarray<EC>& array, std::string& path)
        {
            std::ofstream m_out_file;
            m_out_file.open(path);
            if (m_out_file.is_open())
            {
                dump_csv(m_out_file, array);
                m_out_file.close();
            }
        }

        void read(xarray<EC>& array, std::string& path)
        {
            std::ifstream m_in_file;
            m_in_file.open(path);
            if (m_in_file.is_open())
            {
                array = load_csv<EC>(m_in_file);
                m_in_file.close();
            }
            else
            {
                array = broadcast(0, array.shape());
            }
        }
    };
}

#endif
