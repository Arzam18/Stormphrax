/*
 * Stormphrax, a UCI chess engine
 * Copyright (C) 2026 Ciekce
 *
 * Scalar reference implementation of the Stormphrax SIMD abstraction.
 * This intentionally preserves the same logical vector widths and memory
 * layout as the NEON implementation while using ordinary C++ operations.
 */

#pragma once

#include "../../types.h"
#include "../../arch.h"
#include "../align.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>

#if defined(SP_SCALAR)

namespace stormphrax::util::simd {
    template <typename T, usize N>
    struct ScalarVector {
        std::array<T, N> lanes{};

        constexpr T& operator[](usize i) { return lanes[i]; }
        constexpr const T& operator[](usize i) const { return lanes[i]; }
    };

    using VectorU8 = ScalarVector<u8, 16>;
    using VectorU16 = ScalarVector<u16, 8>;
    using VectorI8 = ScalarVector<i8, 16>;
    using VectorI16 = ScalarVector<i16, 8>;
    using VectorI32 = ScalarVector<i32, 4>;

    constexpr std::uintptr_t kAlignment = 16;
    constexpr bool kPackNonSequential = false;
    constexpr usize kPackGrouping = 1;
    constexpr std::array kPackOrdering = {0};

    namespace detail {
        template <typename T>
        constexpr T clampValue(T x, T lo, T hi) {
            return x < lo ? lo : (x > hi ? hi : x);
        }

        template <typename T>
        constexpr T arithmeticShiftRight(T x, i32 shift) {
            if (shift <= 0) return static_cast<T>(x << (-shift));
            if (shift >= static_cast<i32>(sizeof(T) * 8)) return x < 0 ? T(-1) : T(0);
            return static_cast<T>(x >> shift);
        }

        template <typename T>
        constexpr T saturateToI16(i64 x) {
            return static_cast<T>(clampValue<i64>(
                x, std::numeric_limits<i16>::min(), std::numeric_limits<i16>::max()));
        }

        template <typename T>
        constexpr T saturateToU8(i64 x) {
            return static_cast<T>(clampValue<i64>(x, 0, 255));
        }

        template <typename T>
        constexpr T saturateToU16(i64 x) {
            return static_cast<T>(clampValue<i64>(x, 0, 65535));
        }
    }

    namespace impl {
        template <typename V>
        constexpr V zero() {
            return {};
        }

        inline VectorU8 zeroU8() { return {}; }
        inline VectorU16 zeroU16() { return {}; }
        inline VectorI8 zeroI8() { return {}; }
        inline VectorI16 zeroI16() { return {}; }
        inline VectorI32 zeroI32() { return {}; }

        inline VectorI8 set1I8(i8 v) {
            VectorI8 r; r.lanes.fill(v); return r;
        }
        inline VectorI16 set1I16(i16 v) {
            VectorI16 r; r.lanes.fill(v); return r;
        }
        inline VectorI32 set1I32(i32 v) {
            VectorI32 r; r.lanes.fill(v); return r;
        }

        inline VectorU8 loadU8(const void* ptr) {
            assert(isAligned<kAlignment>(ptr));
            VectorU8 r;
            const auto* p = static_cast<const u8*>(ptr);
            for (usize i = 0; i < 16; ++i) r[i] = p[i];
            return r;
        }
        inline VectorI8 loadI8(const void* ptr) {
            assert(isAligned<kAlignment>(ptr));
            VectorI8 r;
            const auto* p = static_cast<const i8*>(ptr);
            for (usize i = 0; i < 16; ++i) r[i] = p[i];
            return r;
        }
        inline VectorI16 loadI16(const void* ptr) {
            assert(isAligned<kAlignment>(ptr));
            VectorI16 r;
            const auto* p = static_cast<const i16*>(ptr);
            for (usize i = 0; i < 8; ++i) r[i] = p[i];
            return r;
        }
        inline VectorI32 loadI32(const void* ptr) {
            assert(isAligned<kAlignment>(ptr));
            VectorI32 r;
            const auto* p = static_cast<const i32*>(ptr);
            for (usize i = 0; i < 4; ++i) r[i] = p[i];
            return r;
        }

        inline void storeU8(void* ptr, VectorU8 v) {
            assert(isAligned<kAlignment>(ptr));
            auto* p = static_cast<u8*>(ptr);
            for (usize i = 0; i < 16; ++i) p[i] = v[i];
        }
        inline void storeI8(void* ptr, VectorI8 v) {
            assert(isAligned<kAlignment>(ptr));
            auto* p = static_cast<i8*>(ptr);
            for (usize i = 0; i < 16; ++i) p[i] = v[i];
        }
        inline void storeI16(void* ptr, VectorI16 v) {
            assert(isAligned<kAlignment>(ptr));
            auto* p = static_cast<i16*>(ptr);
            for (usize i = 0; i < 8; ++i) p[i] = v[i];
        }
        inline void storeI32(void* ptr, VectorI32 v) {
            assert(isAligned<kAlignment>(ptr));
            auto* p = static_cast<i32*>(ptr);
            for (usize i = 0; i < 4; ++i) p[i] = v[i];
        }

        inline VectorI16 widenLoadI8ToI16(const void* ptr) {
            const auto* p = static_cast<const i8*>(ptr);
            VectorI16 r;
            for (usize i = 0; i < 8; ++i) r[i] = p[i];
            return r;
        }

#define SP_SCALAR_BIN_INT8(Name, OP) \
        inline VectorI8 Name##I8(VectorI8 a, VectorI8 b) { \
            VectorI8 r; for (usize i=0;i<16;++i) r[i]=static_cast<i8>(a[i] OP b[i]); return r; \
        }
#define SP_SCALAR_BIN_INT16(Name, OP) \
        inline VectorI16 Name##I16(VectorI16 a, VectorI16 b) { \
            VectorI16 r; for (usize i=0;i<8;++i) r[i]=static_cast<i16>(a[i] OP b[i]); return r; \
        }
#define SP_SCALAR_BIN_INT32(Name, OP) \
        inline VectorI32 Name##I32(VectorI32 a, VectorI32 b) { \
            VectorI32 r; for (usize i=0;i<4;++i) r[i]=static_cast<i32>(a[i] OP b[i]); return r; \
        }

        SP_SCALAR_BIN_INT8(add, +)
        SP_SCALAR_BIN_INT8(sub, -)
        inline VectorI8 minI8(VectorI8 a, VectorI8 b) {
            VectorI8 r; for (usize i=0;i<16;++i) r[i]=std::min(a[i], b[i]); return r;
        }
        inline VectorI8 maxI8(VectorI8 a, VectorI8 b) {
            VectorI8 r; for (usize i=0;i<16;++i) r[i]=std::max(a[i], b[i]); return r;
        }
        SP_SCALAR_BIN_INT16(add, +)
        SP_SCALAR_BIN_INT16(sub, -)
        inline VectorI16 minI16(VectorI16 a, VectorI16 b) {
            VectorI16 r; for (usize i=0;i<8;++i) r[i]=std::min(a[i], b[i]); return r;
        }
        inline VectorI16 maxI16(VectorI16 a, VectorI16 b) {
            VectorI16 r; for (usize i=0;i<8;++i) r[i]=std::max(a[i], b[i]); return r;
        }
        SP_SCALAR_BIN_INT32(add, +)
        SP_SCALAR_BIN_INT32(sub, -)
        inline VectorI32 minI32(VectorI32 a, VectorI32 b) {
            VectorI32 r; for (usize i=0;i<4;++i) r[i]=std::min(a[i], b[i]); return r;
        }
        inline VectorI32 maxI32(VectorI32 a, VectorI32 b) {
            VectorI32 r; for (usize i=0;i<4;++i) r[i]=std::max(a[i], b[i]); return r;
        }
#undef SP_SCALAR_BIN_INT8
#undef SP_SCALAR_BIN_INT16
#undef SP_SCALAR_BIN_INT32

        inline VectorI8 clampI8(VectorI8 v, VectorI8 lo, VectorI8 hi) {
            return maxI8(lo, minI8(v, hi));
        }
        inline VectorI16 clampI16(VectorI16 v, VectorI16 lo, VectorI16 hi) {
            return maxI16(lo, minI16(v, hi));
        }
        inline VectorI32 clampI32(VectorI32 v, VectorI32 lo, VectorI32 hi) {
            return maxI32(lo, minI32(v, hi));
        }

        inline VectorI8 shiftLeftI8(VectorI8 v, i32 s) {
            VectorI8 r;
            for (usize i=0;i<16;++i) r[i]=static_cast<i8>(static_cast<i16>(v[i]) << s);
            return r;
        }
        inline VectorI16 shiftLeftI16(VectorI16 v, i32 s) {
            VectorI16 r;
            for (usize i=0;i<8;++i) r[i]=static_cast<i16>(static_cast<i64>(v[i]) << s);
            return r;
        }
        inline VectorI16 shiftRightI16(VectorI16 v, i32 s) {
            VectorI16 r;
            for (usize i=0;i<8;++i) r[i]=detail::arithmeticShiftRight(v[i], s);
            return r;
        }
        inline VectorI32 shiftLeftI32(VectorI32 v, i32 s) {
            VectorI32 r;
            for (usize i=0;i<4;++i) r[i]=static_cast<i32>(static_cast<i64>(v[i]) << s);
            return r;
        }
        inline VectorI32 shiftRightI32(VectorI32 v, i32 s) {
            VectorI32 r;
            for (usize i=0;i<4;++i) r[i]=detail::arithmeticShiftRight(v[i], s);
            return r;
        }

        inline VectorI16 mulLoI16(VectorI16 a, VectorI16 b) {
            VectorI16 r;
            for (usize i=0;i<8;++i) r[i]=static_cast<i16>(static_cast<i32>(a[i])*b[i]);
            return r;
        }
        inline VectorI32 mulLoI32(VectorI32 a, VectorI32 b) {
            VectorI32 r;
            for (usize i=0;i<4;++i) r[i]=static_cast<i32>(static_cast<i64>(a[i])*b[i]);
            return r;
        }

        inline VectorI16 shiftLeftMulHiI16(VectorI16 a, VectorI16 b, i32 shift) {
            VectorI16 r;
            for (usize i=0;i<8;++i) {
                const i64 shifted = static_cast<i64>(a[i]) << (shift - 1);
                const i64 product = shifted * static_cast<i64>(b[i]);
                r[i] = detail::saturateToI16<i16>(product >> 15);
            }
            return r;
        }

        inline VectorI32 mulAddAdjI16(VectorI16 a, VectorI16 b) {
            VectorI32 r;
            for (usize i=0;i<4;++i)
                r[i] = static_cast<i32>(a[i*2])*b[i*2] + static_cast<i32>(a[i*2+1])*b[i*2+1];
            return r;
        }

        inline VectorI32 mulAddAdjAccI16(VectorI32 sum, VectorI16 a, VectorI16 b) {
            const auto p = mulAddAdjI16(a,b);
            return addI32(sum,p);
        }

        inline VectorU8 packUnsignedI16(VectorI16 a, VectorI16 b) {
            VectorU8 r;
            for (usize i=0;i<8;++i) r[i]=detail::saturateToU8<u8>(a[i]);
            for (usize i=0;i<8;++i) r[i+8]=detail::saturateToU8<u8>(b[i]);
            return r;
        }

        inline VectorU16 packUnsignedI32(VectorI32 a, VectorI32 b) {
            VectorU16 r;
            for (usize i=0;i<4;++i) r[i]=detail::saturateToU16<u16>(a[i]);
            for (usize i=0;i<4;++i) r[i+4]=detail::saturateToU16<u16>(b[i]);
            return r;
        }

        inline i32 hsumI32(VectorI32 v) {
            return v[0]+v[1]+v[2]+v[3];
        }

        inline VectorI32 dpbusdI32(VectorI32 sum, VectorU8 u, VectorI8 i) {
            VectorI32 r = sum;
            for (usize lane=0; lane<4; ++lane) {
                i32 dot = 0;
                for (usize j=0; j<4; ++j)
                    dot += static_cast<i32>(u[lane*4+j]) * static_cast<i32>(i[lane*4+j]);
                r[lane] += dot;
            }
            return r;
        }

        inline u32 nonzeroMaskU8(VectorU8 v) {
            u32 mask = 0;
            for (usize lane=0; lane<4; ++lane) {
                if (v[lane*4] || v[lane*4+1] || v[lane*4+2] || v[lane*4+3])
                    mask |= (1u << lane);
            }
            return mask;
        }
    }
}

#endif
