#include <vector>
#include <array>
#include "xarray.hpp"

namespace xt
{
    template <class chunk_type>
    class xchunked_array: public xt::xaccessible<xchunked_array<chunk_type>>
    {
    public:

        using const_reference = typename chunk_type::const_reference;
        using reference = typename chunk_type::reference;

        template <class... Idxs>
        inline const_reference operator()(Idxs... idxs) const
        {
            auto ci = get_chunk_and_indexes(idxs...);
            return ci.first.element(ci.second.cbegin(), ci.second.cend());
        }

        template <class... Idxs>
        inline reference operator()(Idxs... idxs)
        {
            auto ci = get_chunk_and_indexes(idxs...);
            return ci.first.element(ci.second.cbegin(), ci.second.cend());
        }

        xchunked_array(std::vector<size_t> shape, std::vector<size_t> chunk_shape):
            m_shape(shape),
            m_chunk_shape(chunk_shape)
        {
            std::vector<size_t> shape_chunk(shape.size());
            size_t di = 0;
            for (auto s: shape)
            {
                size_t chunk_nb = s / chunk_shape[di];
                if (s % chunk_shape[di] > 0)
                    chunk_nb += 1;  // edge chunk
                shape_chunk[di] = chunk_nb;
                di++;
            }
            m_chunks.resize(shape_chunk);
            for (auto& c: m_chunks)
                c.resize(chunk_shape);
        }

        reference operator[](const xindex& index)
        {
            reference el = element(index.cbegin(), index.cend());
            return el;
        }

        const_reference operator[](const xindex& index) const
        {
            const_reference const_el = element(index.cbegin(), index.cend());
            return const_el;
        }

    private:

        xt::xarray<chunk_type> m_chunks;
        std::vector<size_t> m_shape;
        std::vector<size_t> m_chunk_shape;

        template <class... Idxs>
        inline std::pair<chunk_type, std::array<size_t, sizeof...(Idxs)>> get_chunk_and_indexes(Idxs... idxs) const
        {
            auto chunk_indexes_packed = get_chunk_indexes(std::make_index_sequence<sizeof...(Idxs)>(), idxs...);
            auto chunk_indexes = unpack(chunk_indexes_packed);
            auto indexes_of_chunk = chunk_indexes.first;
            auto indexes_in_chunk = chunk_indexes.second;
            auto chunk = m_chunks.element(indexes_of_chunk.cbegin(), indexes_of_chunk.cend());
            return std::make_pair(chunk, indexes_in_chunk);
        }

        template <class Dim, class Idx>
        std::pair<size_t, size_t> get_chunk_indexes_in_dimension(Dim dim, Idx idx) const
        {
            size_t index_of_chunk = idx / m_chunk_shape[dim];
            size_t index_in_chunk = idx - index_of_chunk * m_chunk_shape[dim];
            return std::make_pair(index_of_chunk, index_in_chunk);
        }

        template <size_t... dims, class... Idxs>
        std::array<std::pair<size_t, size_t>, sizeof...(Idxs)>
        get_chunk_indexes(std::index_sequence<dims...>, Idxs... idxs) const
        {
            std::array<std::pair<size_t, size_t>, sizeof...(Idxs)> chunk_indexes = {{get_chunk_indexes_in_dimension(dims, idxs)...}};
            return chunk_indexes;
        }

        template <class T, std::size_t N>
        std::pair<std::array<size_t, N>, std::array<size_t, N>> unpack(std::array<T, N> &arr) const
        {
            std::array<size_t, N> arr0;
            std::array<size_t, N> arr1;
            for (size_t i = 0; i < N; ++i)
            {
                arr0[i] = std::get<0>(arr[i]);
                arr1[i] = std::get<1>(arr[i]);
            }
            return std::make_pair(arr0, arr1);
        }

        template <class It>
        inline std::pair<chunk_type, std::vector<size_t>> get_chunk_and_indexes_dynamic(It first, It last) const
        {
            std::vector<size_t> indexes_of_chunk;
            std::vector<size_t> indexes_in_chunk;
            std::pair<size_t, size_t> chunk_index;
            int dim = 0;
            for (auto it = first; it != last; ++it)
            {
                chunk_index = get_chunk_indexes_in_dimension(dim, *it);
                indexes_of_chunk.push_back(chunk_index.first);
                indexes_in_chunk.push_back(chunk_index.second);
                dim++;
            }
            chunk_type chunk = m_chunks.element(indexes_of_chunk.begin(), indexes_of_chunk.end());
            return std::make_pair(chunk, indexes_in_chunk);
        }

        template <class It>
        inline reference element(It first, It last)
        {
            auto ci = get_chunk_and_indexes_dynamic(first, last);
            return ci.first.element(ci.second.begin(), ci.second.end());
        }

        template <class It>
        inline const_reference element(It first, It last) const
        {
            auto ci = get_chunk_and_indexes_dynamic(first, last);
            return ci.first.element(ci.second.begin(), ci.second.end());
        }
    };

    template <class chunk_type>
    struct xcontainer_inner_types<xchunked_array<chunk_type>>
    {
        using temporary_type = xarray<chunk_type>;
        using const_reference = typename chunk_type::const_reference;
        using reference = typename chunk_type::reference;
        using size_type = std::size_t;
    };
}
