#ifndef XTENSOR_CHUNK_STORE_MANAGER_HPP
#define XTENSOR_CHUNK_STORE_MANAGER_HPP

#include <vector>
#include <array>

#include "xarray.hpp"
#include "xcsv.hpp"
#include "xio.hpp"

namespace xt
{

    /***************************
     * xindex_path declaration *
     ***************************/

    class xindex_path
    {
    public:
        void set_directory(const std::string& directory);
        template <class I>
        void index_to_path(I, I, std::string&);

    private:
        std::string m_directory;
    };

    /************************************
     * xchunk_store_manager declaration *
     ************************************/

    template <class EC, class IP>
    class xchunk_store_manager;

    template <class EC, class IP>
    struct xcontainer_inner_types<xchunk_store_manager<EC, IP>>
    {
        using storage_type = EC;
        using reference = EC&;
        using const_reference = const EC&;
        using size_type = std::size_t;
        using temporary_type = xchunk_store_manager<EC, IP>;
    };

    template <class EC, class IP>
    struct xiterable_inner_types<xchunk_store_manager<EC, IP>>
    {
        using inner_shape_type = std::vector<std::size_t>;
        using stepper = xindexed_stepper<xchunk_store_manager<EC, IP>, false>;
        using const_stepper = xindexed_stepper<xchunk_store_manager<EC, IP>, true>;
    };

    template <class EC, class IP = xindex_path>
    class xchunk_store_manager: public xaccessible<xchunk_store_manager<EC, IP>>,
                                public xiterable<xchunk_store_manager<EC, IP>>
    {
    public:

        using self_type = xchunk_store_manager<EC, IP>;
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

        template <class S>
        xchunk_store_manager(S&& shape, S&& chunk_shape, const std::string& directory, std::size_t pool_size);
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

        std::size_t size();

        IP& get_index_path();
        void flush();

        template <class C>
        void configure_format(C& config);

        template <class I>
        reference map_file_array(I first, I last);

        template <class I>
        const_reference map_file_array(I first, I last) const;

    private:

        template <class... Idxs>
        std::array<std::size_t, sizeof...(Idxs)> get_indexes(Idxs... idxs) const;

        using chunk_pool_type = std::vector<EC>;
        using index_pool_type = std::vector<shape_type>;

        shape_type m_shape;
        chunk_pool_type m_chunk_pool;
        index_pool_type m_index_pool;
        std::size_t m_unload_index;
        IP m_index_path;
    };

    /******************************
     * xindex_path implementation *
     ******************************/

    inline void xindex_path::set_directory(const std::string& directory)
    {
        m_directory = directory;
        if (m_directory.back() != '/')
        {
            m_directory.push_back('/');
        }
    }

    template <class I>
    void xindex_path::index_to_path(I first, I last, std::string& path)
    {
        std::string fname;
        for (auto it = first; it != last; ++it)
        {
            if (!fname.empty())
            {
                fname.push_back('.');
            }
            fname.append(std::to_string(*it));
        }
        path = m_directory + fname;
    }

    /***************************************
     * xchunk_store_manager implementation *
     ***************************************/

    template <class EC, class IP>
    template <class S>
    inline xchunk_store_manager<EC, IP>::xchunk_store_manager(S&& shape,
                                                              S&& chunk_shape,
                                                              const std::string& directory,
                                                              std::size_t pool_size)
        : m_shape(shape)
        , m_chunk_pool(pool_size)
        , m_index_pool(pool_size)
        , m_unload_index(0u)
    {
        // resize the pool chunks
        for (auto& chunk: m_chunk_pool)
        {
            chunk.resize(chunk_shape);
            chunk.ignore_empty_path(true);
        }
        m_index_path.set_directory(directory);
    }

    template <class EC, class IP>
    inline auto xchunk_store_manager<EC, IP>::shape() const noexcept -> const shape_type&
    {
        return m_shape;
    }

    template <class EC, class IP>
    template <class... Idxs>
    inline auto xchunk_store_manager<EC, IP>::operator()(Idxs... idxs) -> reference
    {
        auto index = get_indexes(idxs...);
        return map_file_array(index.cbegin(), index.cend());
    }

    template <class EC, class IP>
    template <class... Idxs>
    inline auto xchunk_store_manager<EC, IP>::operator()(Idxs... idxs) const -> const_reference
    {
        auto index = get_indexes(idxs...);
        return map_file_array(index.cbegin(), index.cend());
    }

    template <class EC, class IP>
    template <class It>
    inline auto xchunk_store_manager<EC, IP>::element(It first, It last) -> reference
    {
        return map_file_array(first, last);
    }

    template <class EC, class IP>
    template <class It>
    inline auto xchunk_store_manager<EC, IP>::element(It first, It last) const -> const_reference
    {
        return map_file_array(first, last);
    }

    template <class EC, class IP>
    template <class O>
    inline auto xchunk_store_manager<EC, IP>::stepper_begin(const O& shape) noexcept -> stepper
    {
        size_type offset = shape.size() - this->dimension();
        return stepper(this, offset);
    }

    template <class EC, class IP>
    template <class O>
    inline auto xchunk_store_manager<EC, IP>::stepper_end(const O& shape, layout_type) noexcept -> stepper
    {
        size_type offset = shape.size() - this->dimension();
        return stepper(this, offset, true);
    }

    template <class EC, class IP>
    template <class O>
    inline auto xchunk_store_manager<EC, IP>::stepper_begin(const O& shape) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - this->dimension();
        return const_stepper(this, offset);
    }

    template <class EC, class IP>
    template <class O>
    inline auto xchunk_store_manager<EC, IP>::stepper_end(const O& shape, layout_type) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - this->dimension();
        return const_stepper(this, offset, true);
    }

    template <class EC, class IP>
    template <class S>
    inline void xchunk_store_manager<EC, IP>::resize(S&& shape)
    {
        // don't resize according to total number of chunks
        // instead the pool manages a number of in-memory chunks
        m_shape = shape;
    }

    template <class EC, class IP>
    inline std::size_t xchunk_store_manager<EC, IP>::size()
    {
        return compute_size(m_shape);
    }

    template <class EC, class IP>
    inline void xchunk_store_manager<EC, IP>::flush()
    {
        for (auto& chunk: m_chunk_pool)
        {
            chunk.flush();
        }
    }

    template <class EC, class IP>
    template <class C>
    void xchunk_store_manager<EC, IP>::configure_format(C& config)
    {
        for (auto& chunk: m_chunk_pool)
        {
            chunk.configure_format(config);
        }
    }

    template <class EC, class IP>
    IP& xchunk_store_manager<EC, IP>::get_index_path()
    {
        return m_index_path;
    }

    template <class EC, class IP>
    template <class I>
    inline auto xchunk_store_manager<EC, IP>::map_file_array(I first, I last) -> reference
    {
        std::string path;
        m_index_path.index_to_path(first, last, path);
        if (first == last)
        {
            return m_chunk_pool[0];
        }
        else
        {
            // check if the chunk is already loaded in memory
            const auto it1 = std::find_if(m_index_pool.cbegin(), m_index_pool.cend(), [first, last](const auto& v)
                { return std::equal(v.cbegin(), v.cend(), first, last); });
            std::size_t i;
            if (it1 != m_index_pool.cend())
            {
                i = static_cast<std::size_t>(std::distance(m_index_pool.cbegin(), it1));
                return m_chunk_pool[i];
            }
            // if not, find a free chunk in the pool
            std::vector<std::size_t> empty_index;
            const auto it2 = std::find(m_index_pool.cbegin(), m_index_pool.cend(), empty_index);
            if (it2 != m_index_pool.cend())
            {
                i = static_cast<std::size_t>(std::distance(m_index_pool.cbegin(), it2));
                m_chunk_pool[i].set_path(path);
                m_index_pool[i].resize(static_cast<size_t>(std::distance(first, last)));
                std::copy(first, last, m_index_pool[i].begin());
                return m_chunk_pool[i];
            }
            // no free chunk, take one (which will thus be unloaded)
            // fairness is guaranteed through the use of a walking index
            m_chunk_pool[m_unload_index].set_path(path);
            m_index_pool[m_unload_index].resize(static_cast<size_t>(std::distance(first, last)));
            std::copy(first, last, m_index_pool[m_unload_index].begin());
            auto& chunk = m_chunk_pool[m_unload_index];
            m_unload_index = (m_unload_index + 1) % m_index_pool.size();
            return chunk;
        }
    }

    template <class EC, class IP>
    template <class I>
    inline auto xchunk_store_manager<EC, IP>::map_file_array(I first, I last) const -> const_reference
    {
        return const_cast<xchunk_store_manager<EC, IP>*>(this)->map_file_array(first, last);
    }

    template <class EC, class IP>
    template <class... Idxs>
    inline std::array<std::size_t, sizeof...(Idxs)>
    xchunk_store_manager<EC, IP>::get_indexes(Idxs... idxs) const
    {
        std::array<std::size_t, sizeof...(Idxs)> indexes = {{idxs...}};
        return indexes;
    }
}

#endif
