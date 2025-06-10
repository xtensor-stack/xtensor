/***************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <benchmark/benchmark.h>

#include "xtensor/containers/xtensor.hpp"
#include "xtensor/core/xmath.hpp"
#include "xtensor/generators/xrandom.hpp"

namespace xt
{
    namespace
    {
        constexpr std::array<size_t, 2> cContainerAssignShape{2000, 2000};

        template <class Shape>
        auto generateRandomInt16From0To100(Shape&& x)
        {
            return xt::random::randint(x, 0, 100);
        }
    }

    static void Xtensor_Uint16_2000x2000_DivideBy2_StdTransform(benchmark::State& aState)
    {
        xt::xtensor<uint16_t, 2> vInput = generateRandomInt16From0To100(cContainerAssignShape);
        auto vOutput = xt::xtensor<uint16_t, 2>::from_shape(cContainerAssignShape);

        for (auto _ : aState)
        {
            std::transform(
                vInput.begin(),
                vInput.end(),
                vOutput.begin(),
                [](auto&& aInputValue)
                {
                    return aInputValue / 2;
                }
            );
        }
    }

    static void Xtensor_Uint16_2000x2000_DivideBy2_Xtensor(benchmark::State& aState)
    {
        xt::xtensor<uint16_t, 2> vInput = generateRandomInt16From0To100(cContainerAssignShape);
        auto vOutput = xt::xtensor<uint16_t, 2>::from_shape(cContainerAssignShape);

        for (auto _ : aState)
        {
            vOutput = vInput / 2;
        }
    }

    static void Xtensor_Uint16_2000x2000_DivideBy2Double_StdTransform(benchmark::State& aState)
    {
        xt::xtensor<uint16_t, 2> vInput = generateRandomInt16From0To100(cContainerAssignShape);
        auto vOutput = xt::xtensor<uint16_t, 2>::from_shape(cContainerAssignShape);

        for (auto _ : aState)
        {
            std::transform(
                vInput.begin(),
                vInput.end(),
                vOutput.begin(),
                [](auto&& aInputValue)
                {
                    return aInputValue / 2.0;
                }
            );
        }
    }

    static void Xtensor_Uint16_2000x2000_DivideBy2Double_Xtensor(benchmark::State& aState)
    {
        xt::xtensor<uint16_t, 2> vInput = generateRandomInt16From0To100(cContainerAssignShape);
        auto vOutput = xt::xtensor<uint16_t, 2>::from_shape(cContainerAssignShape);

        for (auto _ : aState)
        {
            vOutput = vInput / 2.0;
        }
    }

    static void Xtensor_Uint16_2000x2000_MultiplyBy2_StdTransform(benchmark::State& aState)
    {
        xt::xtensor<uint16_t, 2> vInput = generateRandomInt16From0To100(cContainerAssignShape);
        auto vOutput = xt::xtensor<uint16_t, 2>::from_shape(cContainerAssignShape);

        for (auto _ : aState)
        {
            std::transform(
                vInput.begin(),
                vInput.end(),
                vOutput.begin(),
                [](auto&& aInputValue)
                {
                    return aInputValue * 2;
                }
            );
        }
    }

    static void Xtensor_Uint16_2000x2000_MultiplyBy2_Xtensor(benchmark::State& aState)
    {
        xt::xtensor<uint16_t, 2> vInput = generateRandomInt16From0To100(cContainerAssignShape);
        auto vOutput = xt::xtensor<uint16_t, 2>::from_shape(cContainerAssignShape);

        for (auto _ : aState)
        {
            vOutput = vInput * 2;
        }
    }

    static void Xtensor_Uint16_2000x2000_Maximum_StdTransform(benchmark::State& aState)
    {
        xt::xtensor<uint16_t, 2> vInput1 = generateRandomInt16From0To100(cContainerAssignShape);
        xt::xtensor<uint16_t, 2> vInput2 = generateRandomInt16From0To100(cContainerAssignShape);
        auto vOutput = xt::xtensor<uint16_t, 2>::from_shape(cContainerAssignShape);

        for (auto _ : aState)
        {
            auto vInput2It = vInput2.begin();
            std::transform(
                vInput1.begin(),
                vInput1.end(),
                vOutput.begin(),
                [&vInput2It](auto&& aInput1Value)
                {
                    return std::max(aInput1Value, *vInput2It++);
                }
            );
        }
    }

    static void Xtensor_Uint16_2000x2000_Maximum_Xtensor(benchmark::State& aState)
    {
        xt::xtensor<uint16_t, 2> vInput1 = generateRandomInt16From0To100(cContainerAssignShape);
        xt::xtensor<uint16_t, 2> vInput2 = generateRandomInt16From0To100(cContainerAssignShape);
        auto vOutput = xt::xtensor<uint16_t, 2>::from_shape(cContainerAssignShape);

        for (auto _ : aState)
        {
            vOutput = xt::maximum(vInput1, vInput2);
        }
    }

    BENCHMARK(Xtensor_Uint16_2000x2000_Maximum_Xtensor);
    BENCHMARK(Xtensor_Uint16_2000x2000_Maximum_StdTransform);
    BENCHMARK(Xtensor_Uint16_2000x2000_MultiplyBy2_Xtensor);
    BENCHMARK(Xtensor_Uint16_2000x2000_MultiplyBy2_StdTransform);
    BENCHMARK(Xtensor_Uint16_2000x2000_DivideBy2Double_Xtensor);
    BENCHMARK(Xtensor_Uint16_2000x2000_DivideBy2Double_StdTransform);
}
