// SPDX-License-Identifier: GPL-3.0-or-later
// Clunk — LLVM IR superoptimiser
// Copyright (C) 2025 Clunk contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

/*
 * Clunk Heuristics — utility functions for the evaluation engine.
 */
#include "clunk/Evaluator/EvaluationEngine.h"

namespace clunk::evaluator {

std::string memory_layer_name(MemoryLayer layer) {
    switch (layer) {
        case MemoryLayer::Register:   return "Register";
        case MemoryLayer::L1:         return "L1";
        case MemoryLayer::L2:         return "L2";
        case MemoryLayer::L3:         return "L3";
        case MemoryLayer::DRAM:       return "DRAM";
        case MemoryLayer::GPU_Global: return "GPU_Global";
        case MemoryLayer::GPU_Shared: return "GPU_Shared";
        case MemoryLayer::GPU_Local:  return "GPU_Local";
        case MemoryLayer::NVMe:       return "NVMe";
        case MemoryLayer::Network:    return "Network";
    }
    return "Unknown";
}

} // namespace clunk::evaluator
