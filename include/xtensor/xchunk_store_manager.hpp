#ifndef XTENSOR_FILES_ARRAY_HPP
#define XTENSOR_FILES_ARRAY_HPP

#include <istream>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>

#include "xarray.hpp"
#include "xcsv.hpp"
#include "xio.hpp"

namespace xt
{
    template <class S>
    void reopen(S& stream, const std::string& path)
    {
        if (stream.is_open())
            stream.close();
        stream.open(path);
    }

    template <class EC, class SIN, class SOUT>
    class xfiles_array;

    template <class EC, class SIN, class SOUT>
    struct xcontainer_inner_types<xfiles_array<EC, SIN, SOUT>>
    {
        using const_reference = const EC&;
        using reference = EC&;
        using size_type = std::size_t;
        using storage_type = EC;
        using temporary_type = xfiles_array<EC, SIN, SOUT>;
    };

    template <class EC, class SIN, class SOUT>
    struct xiterable_inner_types<xfiles_array<EC, SIN, SOUT>>
    {
        using inner_shape_type = std::vector<size_t>;
        using const_stepper = xindexed_stepper<xfiles_array<EC, SIN, SOUT>, true>;
        using stepper = xindexed_stepper<xfiles_array<EC, SIN, SOUT>, false>;
    };

    template <class EC, class SIN, class SOUT>
    class xfiles_array: public xaccessible<xfiles_array<EC, SIN, SOUT>>,
                        public xiterable<xfiles_array<EC, SIN, SOUT>>
    {
    public:

        using const_reference = const EC&;
        using reference = EC&;
        using self_type = xfiles_array<EC, SIN, SOUT>;
        using iterable_base = xconst_iterable<self_type>;
        using const_stepper = typename iterable_base::const_stepper;
        using stepper = typename iterable_base::stepper;
        using inner_types = xcontainer_inner_types<self_type>;
        using size_type = typename inner_types::size_type;
        using storage_type = typename inner_types::storage_type;
        using value_type = storage_type;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::vector<size_t>;

        template <class O>
        const_stepper stepper_begin(const O& shape) const noexcept;
        template <class O>
        const_stepper stepper_end(const O& shape, layout_type) const noexcept;

        template <class O>
        stepper stepper_begin(const O& shape) noexcept;
        template <class O>
        stepper stepper_end(const O& shape, layout_type) noexcept;

        const shape_type& shape() const
        {
            return m_shape;
        }

        xfiles_array()
        {
            m_array_dirty = false;
        }

        template <class I>
        void map_file_array(I first, I last)
        {
            std::string path;
            for (auto it = first; it != last; ++it)
            {
                if (!path.empty())
                    path.append(".");
                path.append(std::to_string(*it));
            }
            if ((path != m_path) || (path.empty()))
            {
                // accessing a different chunk
                if (m_array_dirty)
                {
                    // chunk was changed, dump it
                    m_array_dirty = false;
                    reopen(m_out_file, m_path);
                    if (m_out_file.is_open())
                        dump_csv(m_out_file, m_array);
                }
                m_path = path;
                // load new chunk
                reopen(m_in_file, path);
                if (m_in_file.is_open())
                    m_array = load_csv<typename EC::value_type>(m_in_file);
                else
                    m_array = broadcast(0, m_array.shape());
                // map file array
                m_file_array = EC(m_array, m_array_dirty);
            }
        }

        template <class... Idxs>
        inline const_reference operator()(Idxs... idxs) const
        {
            auto index = get_indexes(idxs...);
            map_file_array(index.cbegin(), index.cend());
            return m_file_array;
        }

        template <class... Idxs>
        inline reference operator()(Idxs... idxs)
        {
            auto index = get_indexes(idxs...);
            map_file_array(index.cbegin(), index.cend());
            return m_file_array;
        }

        xfiles_array(const xfiles_array&) = default;
        xfiles_array& operator=(const xfiles_array&) = default;

        xfiles_array(xfiles_array&&) = default;
        xfiles_array& operator=(xfiles_array&&) = default;

        reference operator[](const xindex& index)
        {
            map_file_array(index.cbegin(), index.cend());
            return m_file_array;
        }

        const_reference operator[](const xindex& index) const
        {
            map_file_array(index.cbegin(), index.cend());
            return m_file_array;
        }

        template <class It>
        inline reference element(It first, It last)
        {
            map_file_array(first, last);
            return m_file_array;
        }

        template <class It>
        inline const_reference element(It first, It last) const
        {
            map_file_array(first, last);
            return m_file_array;
        }

        size_type dimension() const
        {
            return shape().size();
        }

        template <class S>
        void resize(S& shape)
        {
        }

    private:

        shape_type m_shape;
        EC m_file_array;
        xarray<typename EC::value_type> m_array;
        bool m_array_dirty;
        SIN m_in_file;
        SOUT m_out_file;
        std::string m_path;

        template <class... Idxs>
        inline std::array<size_t, sizeof...(Idxs)> get_indexes(Idxs... idxs) const
        {
            std::array<size_t, sizeof...(Idxs)> indexes = {{idxs...}};
            return indexes;
        }
    };

    template <class EC, class SIN, class SOUT>
    template <class O>
    inline auto xfiles_array<EC, SIN, SOUT>::stepper_begin(const O& shape) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - this->dimension();
        return const_stepper(this, offset);
    }

    template <class EC, class SIN, class SOUT>
    template <class O>
    inline auto xfiles_array<EC, SIN, SOUT>::stepper_end(const O& shape, layout_type) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - this->dimension();
        return const_stepper(this, offset, true);
    }

    template <class EC, class SIN, class SOUT>
    template <class O>
    inline auto xfiles_array<EC, SIN, SOUT>::stepper_begin(const O& shape) noexcept -> stepper
    {
        size_type offset = shape.size() - this->dimension();
        return stepper(this, offset);
    }

    template <class EC, class SIN, class SOUT>
    template <class O>
    inline auto xfiles_array<EC, SIN, SOUT>::stepper_end(const O& shape, layout_type) noexcept -> stepper
    {
        size_type offset = shape.size() - this->dimension();
        return stepper(this, offset, true);
    }
}

#endif