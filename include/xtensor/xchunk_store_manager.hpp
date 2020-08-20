#ifndef XTENSOR_CHUNK_STORE_MANAGER_HPP
#define XTENSOR_CHUNK_STORE_MANAGER_HPP

#include <vector>
#include <array>

#include "xarray.hpp"
#include "xcsv.hpp"
#include "xio.hpp"

namespace xt
{
    
    /************************************
     * xchunk_store_manager declaration *
     ************************************/

    template <class EC>
    class xchunk_store_manager;

    template <class EC>
    struct xcontainer_inner_types<xchunk_store_manager<EC>>
    {
        using storage_type = EC;
        using reference = EC&;
        using const_reference = const EC&;
        using size_type = std::size_t;
        using temporary_type = xchunk_store_manager<EC>;
    };

    template <class EC>
    struct xiterable_inner_types<xchunk_store_manager<EC>>
    {
        using inner_shape_type = std::vector<std::size_t>;
        using stepper = xindexed_stepper<xchunk_store_manager<EC>, false>;
        using const_stepper = xindexed_stepper<xchunk_store_manager<EC>, true>;
    };

    template <class EC>
    class xchunk_store_manager: public xaccessible<xchunk_store_manager<EC>>,
                                public xiterable<xchunk_store_manager<EC>>
    {
    public:

        using self_type = xchunk_store_manager<EC>;
        using inner_types = xcontainer_inner_types<self_type>;
        using storage_type = typename inner_types::storage_type;
        using value_type = storage_type;
        using reference = EC&;
        using const_reference = const EC&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = typename inner_types::size_type;
        using difference_type = std::ptrdiff_t;
        using iterable_base = xconst_iterable<self_type>;
        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;
        using shape_type = typename iterable_base::inner_shape_type;

        xchunk_store_manager();
        ~xchunk_store_manager() = default;

        xchunk_store_manager(const xchunk_store_manager&) = default;
        xchunk_store_manager& operator=(const xchunk_store_manager&) = default;

        xchunk_store_manager(xchunk_store_manager&&) = default;
        xchunk_store_manager& operator=(xchunk_store_manager&&) = default;

        const shape_type& shape() const noexcept;

        template <class... Idxs>
        reference operator()(Idxs... idxs);

        template <class... Idxs>
        const_reference operator()(Idxs... idxs) const;

        template <class It>
        reference element(It first, It last);

        template <class It>
        const_reference element(It first, It last) const;

        template <class O>
        stepper stepper_begin(const O& shape) noexcept;
        template <class O>
        stepper stepper_end(const O& shape, layout_type) noexcept;
        
        template <class O>
        const_stepper stepper_begin(const O& shape) const noexcept;
        template <class O>
        const_stepper stepper_end(const O& shape, layout_type) const noexcept;

        template <class S>
        void resize(S&& shape);

        void set_pool_size(std::size_t n);
        void flush();

        template <class I>
        reference map_file_array(I first, I last);

    private:

        template <class... Idxs>
        std::array<std::size_t, sizeof...(Idxs)> get_indexes(Idxs... idxs) const;

        using chunk_pool_type = std::vector<EC>;
        using index_pool_type = std::vector<shape_type>;

        shape_type m_shape;
        chunk_pool_type m_chunk_pool;
        index_pool_type m_index_pool;
        std::size_t m_unload_index;
    };

    /***************************************
     * xchunk_store_manager implementation *
     ***************************************/

    template <class EC>
    inline xchunk_store_manager<EC>::xchunk_store_manager()
        : m_shape()
        // default pool size is 1
        // so that first chunk is always resized to the chunk shape
        , m_chunk_pool(1u)
        , m_index_pool(1u)
        , m_unload_index(0u)
    {
    }

    template <class EC>
    inline auto xchunk_store_manager<EC>::shape() const noexcept -> const shape_type&
    {
        return m_shape;
    }

    template <class EC>
    template <class... Idxs>
    inline auto xchunk_store_manager<EC>::operator()(Idxs... idxs) -> reference
    {
        auto index = get_indexes(idxs...);
        return map_file_array(index.cbegin(), index.cend());
    }

    template <class EC>
    template <class... Idxs>
    inline auto xchunk_store_manager<EC>::operator()(Idxs... idxs) const -> const_reference
    {
        auto index = get_indexes(idxs...);
        return map_file_array(index.cbegin(), index.cend());
    }

    template <class EC>
    template <class It>
    inline auto xchunk_store_manager<EC>::element(It first, It last) -> reference
    {
        return map_file_array(first, last);
    }

    template <class EC>
    template <class It>
    inline auto xchunk_store_manager<EC>::element(It first, It last) const -> const_reference
    {
        return map_file_array(first, last);
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
    template <class S>
    inline void xchunk_store_manager<EC>::resize(S&&)
    {
        // don't resize according to total number of chunks
        // instead the pool manages a number of in-memory chunks
    }
    
    template <class EC>
    inline void xchunk_store_manager<EC>::set_pool_size(std::size_t n)
    {
        // first chunk always has the correct shape
        // get the shape before resizing the pool
        auto chunk_shape = m_chunk_pool[0].storage().shape();
        m_chunk_pool.resize(n);
        m_index_pool.resize(n);
        m_unload_index = 0;
        // resize the pool chunks
        for (auto& chunk: m_chunk_pool)
        {
            chunk.resize(chunk_shape);
        }
    }

    template <class EC>
    inline void xchunk_store_manager<EC>::flush()
    {
        for (auto& chunk: m_chunk_pool)
        {
            chunk.flush();
        }
    }

    template <class EC>
    template <class I>
    inline auto xchunk_store_manager<EC>::map_file_array(I first, I last) -> reference
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

    template <class EC>
    template <class... Idxs>
    inline std::array<std::size_t, sizeof...(Idxs)>
    xchunk_store_manager<EC>::get_indexes(Idxs... idxs) const
    {
        std::array<std::size_t, sizeof...(Idxs)> indexes = {{idxs...}};
        return indexes;
    }
}

#endif
