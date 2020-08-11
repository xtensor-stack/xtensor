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

        void set_array(xarray<EC>& array)
        {
            m_array = &array;
        }

        void write(std::string& path)
        {
            std::ofstream m_out_file;
            m_out_file.open(path);
            if (m_out_file.is_open())
            {
                dump_csv(m_out_file, *m_array);
                m_out_file.close();
            }
        }

        void read(std::string& path)
        {
            std::ifstream m_in_file;
            m_in_file.open(path);
            if (m_in_file.is_open())
            {
                *m_array = load_csv<EC>(m_in_file);
                m_in_file.close();
            }
            else
                *m_array = broadcast(0, m_array->shape());
        }

    private:

        xarray<EC>* m_array;

    };
}

#endif
