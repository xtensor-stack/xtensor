#ifndef XTENSOR_CHUNK_STORE_MANAGER_HPP
#define XTENSOR_CHUNK_STORE_MANAGER_HPP

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
    template <class EC>
    class xchunk_store_manager;

    template <class EC>
    struct xcontainer_inner_types<xchunk_store_manager<EC>>
    {
        using const_reference = const EC&;
        using reference = EC&;
        using size_type = std::size_t;
        using storage_type = EC;
        using temporary_type = xchunk_store_manager<EC>;
    };

    template <class EC>
    struct xiterable_inner_types<xchunk_store_manager<EC>>
    {
        using inner_shape_type = std::vector<size_t>;
        using const_stepper = xindexed_stepper<xchunk_store_manager<EC>, true>;
        using stepper = xindexed_stepper<xchunk_store_manager<EC>, false>;
    };

    template <class EC>
    class xchunk_store_manager: public xaccessible<xchunk_store_manager<EC>>,
                        public xiterable<xchunk_store_manager<EC>>
    {
    public:

        using const_reference = const EC&;
        using reference = EC&;
        using self_type = xchunk_store_manager<EC>;
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

        xchunk_store_manager()
        {
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
            m_file_array.set_path(path);
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

        xchunk_store_manager(const xchunk_store_manager&) = default;
        xchunk_store_manager& operator=(const xchunk_store_manager&) = default;

        xchunk_store_manager(xchunk_store_manager&&) = default;
        xchunk_store_manager& operator=(xchunk_store_manager&&) = default;

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

        template <class... Idxs>
        inline std::array<size_t, sizeof...(Idxs)> get_indexes(Idxs... idxs) const
        {
            std::array<size_t, sizeof...(Idxs)> indexes = {{idxs...}};
            return indexes;
        }
    };

    template <class EC>
    template <class O>
    inline auto xchunk_store_manager<EC>::stepper_begin(const O& shape) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - this->dimension();
        return const_stepper(this, offset);
    }

    template <class EC>
    template <class O>
    inline auto xchunk_store_manager<EC>::stepper_end(const O& shape, layout_type) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - this->dimension();
        return const_stepper(this, offset, true);
    }

    template <class EC>
    template <class O>
    inline auto xchunk_store_manager<EC>::stepper_begin(const O& shape) noexcept -> stepper
    {
        size_type offset = shape.size() - this->dimension();
        return stepper(this, offset);
    }

    template <class EC>
    template <class O>
    inline auto xchunk_store_manager<EC>::stepper_end(const O& shape, layout_type) noexcept -> stepper
    {
        size_type offset = shape.size() - this->dimension();
        return stepper(this, offset, true);
    }
}

#endif
