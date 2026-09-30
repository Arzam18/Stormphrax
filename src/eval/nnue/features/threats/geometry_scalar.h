/*
 * Stormphrax, a UCI chess engine
 * Copyright (C) 2026 Ciekce
 *
 * Stormphrax is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include "../../../../types.h"

#include <array>
#include <span>
#include <tuple>

namespace stormphrax::eval::nnue::features::threats::geometry {
    // Scalar reference equivalent of the 4x16-byte NEON Vector.
    // Keeping 64 bytes here preserves the exact logical layout used by
    // geometry_neon.h while avoiding architecture-specific intrinsics.
    struct Vector {
        std::array<u8, Squares::kCount> raw{};

        Vector() = default;
        explicit Vector(const std::array<u8, Squares::kCount>& value) : raw(value) {}

        [[nodiscard]] Vector flip() const {
            Vector result;
            for (usize i = 0; i < 16; ++i) {
                result.raw[i]      = raw[32 + i];
                result.raw[16 + i] = raw[48 + i];
                result.raw[32 + i] = raw[i];
                result.raw[48 + i] = raw[16 + i];
            }
            return result;
        }

        [[nodiscard]] Bitrays toMask() const {
            Bitrays result = 0;
            for (usize i = 0; i < Squares::kCount; ++i) {
                if (raw[i] & static_cast<u8>(1u << (i & 7)))
                    result |= Bitrays{1} << i;
            }
            return result;
        }

        [[nodiscard]] static Vector load(const void* ptr) {
            Vector result;
            const auto* p = static_cast<const u8*>(ptr);
            for (usize i = 0; i < Squares::kCount; ++i)
                result.raw[i] = p[i];
            return result;
        }

        template <typename T>
        [[nodiscard]] static Vector cast(const T& v)
            requires(sizeof(T) == sizeof(std::array<u8, Squares::kCount>))
        {
            return load(&v);
        }

        [[nodiscard]] u8& operator[](int index) { return raw[static_cast<usize>(index)]; }
        [[nodiscard]] u8 operator[](int index) const { return raw[static_cast<usize>(index)]; }
    };

    struct Permutation {
        Vector indexes;
        Vector valid;
    };

    [[nodiscard]] inline Permutation permutationFor(Square focus) {
        const auto indexes = Vector::load(kPermutationTable[focus.idx()].data());
        Vector valid;

        for (usize i = 0; i < Squares::kCount; ++i)
            valid.raw[i] = (indexes.raw[i] & 0x80) ? 0 : 0xFF;

        return Permutation{indexes, valid};
    }

    [[nodiscard]] inline std::tuple<Vector, Vector> permuteMailbox(
        const Permutation& permutation,
        Vector mailbox
    ) {
        Vector permuted;
        Vector bits;

        for (usize i = 0; i < Squares::kCount; ++i) {
            const u8 index = permutation.indexes.raw[i];
            const bool valid = permutation.valid.raw[i] != 0;

            if (!valid || index >= Squares::kCount) {
                permuted.raw[i] = 0;
                bits.raw[i] = 0;
                continue;
            }

            const auto piece = mailbox.raw[index];
            permuted.raw[i] = piece;
            bits.raw[i] = kPieceToBitTable[piece & 0x0F];
        }

        return {permuted, bits};
    }

    [[nodiscard]] inline std::tuple<Vector, Vector> permuteMailbox(
        const Permutation& permutation,
        const std::span<const Piece, Squares::kCount> mailbox
    ) {
        return permuteMailbox(permutation, Vector::load(mailbox.data()));
    }

    [[nodiscard]] inline std::tuple<Vector, Vector> permuteMailbox(
        const Permutation& permutation,
        const std::span<const Piece, Squares::kCount> mailbox,
        Square ignore
    ) {
        auto mb = Vector::load(mailbox.data());
        mb.raw[ignore.idx()] = static_cast<u8>(Pieces::kNone.idx());
        return permuteMailbox(permutation, mb);
    }

    [[nodiscard]] inline Bitrays closestOccupied(const Vector& bits) {
        Bitrays occupied = bits.toMask();
        const Bitrays o = occupied | 0x8181818181818181ULL;
        return (o ^ (o - 0x0303030303030303ULL)) & occupied;
    }

    [[nodiscard]] inline Bitrays rayFill(Bitrays br) {
        br = (br + 0x7E7E7E7E7E7E7E7EULL) & 0x8080808080808080ULL;
        return br - (br >> 7);
    }

    [[nodiscard]] inline Bitrays outgoingThreats(Piece piece, Bitrays closest) {
        return kOutgoingThreatsTable[piece.idx()] & closest;
    }

    [[nodiscard]] inline Bitrays incomingAttackers(const Vector& bits, Bitrays closest) {
        Bitrays result = 0;
        for (usize i = 0; i < Squares::kCount; ++i) {
            if ((bits.raw[i] & kIncomingThreatsMask[i]) != 0)
                result |= Bitrays{1} << i;
        }
        return result & closest;
    }

    [[nodiscard]] inline Bitrays incomingSliders(const Vector& bits, Bitrays closest) {
        Bitrays result = 0;
        for (usize i = 0; i < Squares::kCount; ++i) {
            if ((bits.raw[i] & kIncomingSlidersMask[i]) != 0)
                result |= Bitrays{1} << i;
        }
        return result & closest & 0xFEFEFEFEFEFEFEFEULL;
    }
} // namespace stormphrax::eval::nnue::features::threats::geometry
