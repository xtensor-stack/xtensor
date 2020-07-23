#ifndef XTENSOR_FILE_ARRAY_HPP
#define XTENSOR_FILE_ARRAY_HPP

#include "xarray.hpp"

namespace xt
{
    template <class EC>
    class xfile_reference
    {
    public:

        xfile_reference(EC& value, bool& array_dirty)
        {
            m_pvalue = &value;
            m_parray_dirty = &array_dirty;
        }

        operator const EC() const
        {
            return *m_pvalue;
        }

        operator EC()
        {
            return *m_pvalue;
        }

        EC operator=(const EC value)
        {
            if (value != *m_pvalue)
            {
                *m_parray_dirty = true;
                *m_pvalue = value;
            }
            return *m_pvalue;
        }

    private:

        EC* m_pvalue;
        bool* m_parray_dirty;

    };

    template <class EC, class io_handler>
    class xfile_array;

    template <class EC, class io_handler>
    struct xcontainer_inner_types<xfile_array<EC, io_handler>>
    {
        using const_reference = const EC&;
        using reference = EC&;
        using size_type = std::size_t;
        using storage_type = EC;
    };

    template <class EC, class io_handler>
    struct xiterable_inner_types<xfile_array<EC, io_handler>>
    {
        using inner_shape_type = std::vector<size_t>;
        using const_stepper = xindexed_stepper<xfile_array<EC, io_handler>, true>;
        using stepper = xindexed_stepper<xfile_array<EC, io_handler>, false>;
    };

    template <class EC, class io_handler>
    class xfile_array: public xaccessible<xfile_array<EC, io_handler>>,
                       public xiterable<xfile_array<EC, io_handler>>
    {
    public:

        using const_reference = const EC&;
        using reference = xfile_reference<EC>;
        using shape_type = std::vector<size_t>;
        using self_type = xfile_array<EC, io_handler>;
        using inner_types = xcontainer_inner_types<self_type>;
        using size_type = typename inner_types::size_type;
        using storage_type = typename inner_types::storage_type;
        using value_type = storage_type;

        ~xfile_array()
        {
            if (m_array_dirty)
                m_io_handler.write(m_path);
        }

        void set_path(std::string& path)
        {
            if (path != m_path)
            {
                // maybe write to old file
                if (m_array_dirty)
                {
                    m_io_handler.write(m_path);
                    m_array_dirty = false;
                }
                m_path = path;
                // read new file
                m_io_handler.read(path);
            }
        }

        template <class S>
        void resize(S& shape)
        {
            m_array.resize(shape);
            m_array = broadcast(0, shape);
            m_io_handler.set_array(m_array);
        }

        template <class... Idxs>
        inline reference operator()(Idxs... idxs)
        {
            auto index = get_indexes(idxs...);
            return reference(m_array.element(index.cbegin(), index.cend()), m_array_dirty);
        }

        template <class... Idxs>
        inline const_reference operator()(Idxs... idxs) const
        {
            auto index = get_indexes(idxs...);
            return m_array.element(index.cbegin(), index.cend());
        }

        reference operator[](const xindex& index)
        {
            return reference(m_array.element(index.cbegin(), index.cend()), m_array_dirty);
        }

        const_reference operator[](const xindex& index) const
        {
            return m_array.element(index.cbegin(), index.cend());
        }

        template <class It>
        inline reference element(It first, It last)
        {
            return reference(m_array.element(first, last), m_array_dirty);
        }

        template <class It>
        inline const_reference element(It first, It last) const
        {
            return m_array.element(first, last);
        }

    private:

        xarray<EC> m_array;
        bool m_array_dirty;
        io_handler m_io_handler;
        std::string m_path;

        template <class... Idxs>
        inline std::array<size_t, sizeof...(Idxs)> get_indexes(Idxs... idxs) const
        {
            std::array<size_t, sizeof...(Idxs)> indexes = {{idxs...}};
            return indexes;
        }
    };
}

#endif
