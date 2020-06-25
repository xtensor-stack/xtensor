#include <vector>
#include <array>
#include "xarray.hpp"

template <class Chunk_type>
class xchunked_array
{
    public:

        using const_reference = typename Chunk_type::const_reference;

        template <class... Args>
        inline const_reference operator()(Args... args) const
        {
            return access_impl(args...);
        }

        xchunked_array(std::vector<size_t> shape, std::vector<size_t> chunks):
            m_shape(shape),
            m_chunk_shape(chunks)
        {
            std::vector<size_t> shape_chunk;
            size_t di = 0;
            std::cout << "Creating chunked array with shape";
            for (auto s: shape)
            {
                std::cout << " " << s;
                size_t chunk_nb = s / chunks[di];
                if (s % chunks[di] > 0)
                    chunk_nb += 1;  // edge chunk
                shape_chunk.push_back(chunk_nb);
                di++;
            }
            std::cout << ", chunk shape";
            for (auto s: chunks)
                std::cout << " " << s;
            std::cout << std::endl;
            m_chunks.resize(shape_chunk);
        }

    private:

        template <class Idx, class Arg>
        size_t get_index_of_chunk_in_dimension(Idx idx, Arg arg) const
        {
            size_t index_of_chunk = arg / m_chunk_shape[idx];
            return index_of_chunk;
        }

        template <class Idx, class Arg>
        size_t get_index_in_chunk_in_dimension(Idx idx, Arg arg) const
        {
            std::cout << " " << arg;
            size_t index_of_chunk = get_index_of_chunk_in_dimension(idx, arg);
            size_t index_in_chunk = arg - index_of_chunk * m_chunk_shape[idx];
            return index_in_chunk;
        }

        template <size_t... idxs, class... Args>
        std::tuple<std::array<size_t, sizeof...(Args)>, std::array<size_t, sizeof...(Args)>>
        get_chunk_indexes(std::index_sequence<idxs...>, Args... args) const
        {
            std::array<size_t, sizeof...(Args)> indexes_of_chunk = {{get_index_of_chunk_in_dimension(idxs, args)...}};
            std::array<size_t, sizeof...(Args)> indexes_in_chunk = {{get_index_in_chunk_in_dimension(idxs, args)...}};
            return std::make_tuple(indexes_of_chunk, indexes_in_chunk);
        }

        template <class... Args>
        inline const_reference access_impl(Args... args) const
        {
            std::cout << "Accessing chunked array at";
            auto chunk_indexes = get_chunk_indexes(std::make_index_sequence<sizeof...(Args)>(), args...);
            std::cout << std::endl;
            auto indexes_of_chunk(std::get<0>(chunk_indexes));
            auto indexes_in_chunk(std::get<1>(chunk_indexes));
            Chunk_type chunk = m_chunks.element(indexes_of_chunk.cbegin(), indexes_of_chunk.cend());
            const_reference val = chunk.element(indexes_in_chunk.cbegin(), indexes_in_chunk.cend());
            int di = 0;
            for (auto index_of_chunk: indexes_of_chunk)
            {
                auto index_in_chunk = indexes_in_chunk[di];
                std::cout << "Dimension " << di << ": chunk " << index_of_chunk << " at " << index_in_chunk << std::endl;
                di++;
            }
            return val;
        }

        xt::xarray<Chunk_type> m_chunks;
        std::vector<size_t> m_shape;
        std::vector<size_t> m_chunk_shape;
};
