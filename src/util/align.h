/*
 * Stormphrax, a UCI chess engine
 * Copyright (C) 2025 Ciekce
 *
 * Stormphrax is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Stormphrax is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Stormphrax. If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include "../types.h"

#include <cstdlib>


#ifdef __ANDROID__

#include <cstdint>
#include <new>

inline void* aligned_alloc(size_t alignment, size_t size) {
    if (alignment < sizeof(void*)) {
        alignment = sizeof(void*);
    }

    size_t padding = alignment - 1 + sizeof(void*);
    void* p1 = std::malloc(size + padding);
    if (p1 == nullptr) {
        return nullptr;
    }

    void* p2 = reinterpret_cast<void*>(
        (reinterpret_cast<uintptr_t>(p1) + padding) & ~(alignment - 1)
    );
    static_cast<void**>(p2)[-1] = p1;

    return p2;
}

inline void aligned_free(void* ptr) {
    if (ptr != nullptr) {
        std::free(static_cast<void**>(ptr)[-1]);
    }
}

#endif

namespace stormphrax::util {
    template <std::uintptr_t kAlignment, typename T = void>
    constexpr bool isAligned(const T* ptr) {
        return (reinterpret_cast<std::uintptr_t>(ptr) % kAlignment) == 0;
    }

    template <typename T>
    inline T* alignedAlloc(usize alignment, usize count) {
        const auto size = count * sizeof(T);

#ifdef _WIN32
        return static_cast<T*>(_aligned_malloc(size, alignment));
#else
        #ifdef __ANDROID__
	return static_cast<T*>(aligned_alloc(alignment, size));
	    #else
	return static_cast<T*>(std::aligned_alloc(alignment, size));
        #endif
#endif
    }

    inline void alignedFree(void* ptr) {
        if (!ptr) {
            return;
        }

#ifdef _WIN32
        _aligned_free(ptr);
#else
        #ifdef __ANDROID__
	aligned_free(ptr);
	#else
	std::free(ptr);
    #endif
#endif
    }
} // namespace stormphrax::util
