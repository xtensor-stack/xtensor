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
        using inner_shape_type = std::vector<std::size_t>;
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
        using shape_type = std::vector<std::size_t>;

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
            // default pool size is 1
            // so that first chunk is always resized to the chunk shape
            m_chunk_pool.resize(1);
            m_index_pool.resize(1);
            m_unload_index = 0;
        }

        void set_pool_size(std::size_t n)
        {
            // first chunk always has the correct shape
            // get the shape before resizing the pool
            const auto& chunk_shape = m_chunk_pool[0].array().shape();
            m_chunk_pool.resize(n);
            m_index_pool.resize(n);
            m_unload_index = 0;
            // resize the pool chunks
            for (auto& chunk: m_chunk_pool)
            {
                chunk.resize(chunk_shape);
            }
        }

        void flush()
        {
            for (auto& chunk: m_chunk_pool)
            {
                chunk.flush();
            }
        }

        template <class I>
        EC& map_file_array(I first, I last)
        {
            std::string path;
            std::vector<std::size_t> index;
            for (auto it = first; it != last; ++it)
            {
                if (!path.empty())
                {
                    path.append(".");
                }
                path.append(std::to_string(*it));
                index.push_back(*it);
            }
            if (index.empty())
            {
                return m_chunk_pool[0];
            }
            else
            {
                // check if the chunk is already loaded in memory
                const auto it1 = std::find(m_index_pool.cbegin(), m_index_pool.cend(), index);
                std::size_t i;
                if (it1 != m_index_pool.cend())
                {
                    i = std::distance(m_index_pool.cbegin(), it1);
                    return m_chunk_pool[i];
                }
                // if not, find a free chunk in the pool
                std::vector<std::size_t> empty_index;
                const auto it2 = std::find(m_index_pool.cbegin(), m_index_pool.cend(), empty_index);
                if (it2 != m_index_pool.cend())
                {
                    i = std::distance(m_index_pool.cbegin(), it2);
                    m_chunk_pool[i].set_path(path);
                    m_index_pool[i] = index;
                    return m_chunk_pool[i];
                }
                // no free chunk, take one (which will thus be unloaded)
                // fairness is guaranteed through the use of a walking index
                m_chunk_pool[m_unload_index].set_path(path);
                m_index_pool[m_unload_index] = index;
                auto& chunk = m_chunk_pool[m_unload_index];
                m_unload_index = (m_unload_index + 1) % m_index_pool.size();
                return chunk;
            }
        }

        template <class... Idxs>
        inline const_reference operator()(Idxs... idxs) const
        {
            auto index = get_indexes(idxs...);
            return map_file_array(index.cbegin(), index.cend());
        }

        template <class... Idxs>
        inline reference operator()(Idxs... idxs)
        {
            auto index = get_indexes(idxs...);
            return map_file_array(index.cbegin(), index.cend());
        }

        xchunk_store_manager(const xchunk_store_manager&) = default;
        xchunk_store_manager& operator=(const xchunk_store_manager&) = default;

        xchunk_store_manager(xchunk_store_manager&&) = default;
        xchunk_store_manager& operator=(xchunk_store_manager&&) = default;

        reference operator[](const xindex& index)
        {
            return map_file_array(index.cbegin(), index.cend());
        }

        const_reference operator[](const xindex& index) const
        {
            return map_file_array(index.cbegin(), index.cend());
        }

        template <class It>
        inline reference element(It first, It last)
        {
            return map_file_array(first, last);
        }

        template <class It>
        inline const_reference element(It first, It last) const
        {
            return map_file_array(first, last);
        }

        size_type dimension() const
        {
            return shape().size();
        }

        template <class S>
        void resize(S& shape)
        {
            // don't resize according to total number of chunks
            // instead the pool manages a number of in-memory chunks
        }

    private:

        shape_type m_shape;
        std::vector<EC> m_chunk_pool;
        std::vector<std::vector<std::size_t>> m_index_pool;
        std::size_t m_unload_index;

        template <class... Idxs>
        inline std::array<std::size_t, sizeof...(Idxs)> get_indexes(Idxs... idxs) const
        {
            std::array<std::size_t, sizeof...(Idxs)> indexes = {{idxs...}};
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
