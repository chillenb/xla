/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/runtime/buffer_use.h"

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"

namespace xla {

BufferUse::ReadWriteSet::ReadWriteSet() = default;

void BufferUse::ReadWriteSet::Add(BufferUse use) {
  switch (use.access()) {
    case BufferUse::kRead:
      AddRead(use.slice());
      break;
    case BufferUse::kWrite:
      AddWrite(use.slice());
      break;
  }
}

void BufferUse::ReadWriteSet::AddRead(BufferAllocation::Slice slice) {
  read_.insert(slice);
}

void BufferUse::ReadWriteSet::AddWrite(BufferAllocation::Slice slice) {
  write_.insert(slice);
}

void BufferUse::ReadWriteSet::AddAll(absl::Span<const BufferUse> uses) {
  for (const auto& use : uses) Add(use);
}

bool BufferUse::ReadWriteSet::HasConflicts(const BufferUse& use) const {
  // Returns true if slice overlaps with any of the slices in read set.
  auto read_overlap = [&](const BufferAllocation::Slice& slice) {
    if (read_.contains(slice)) return true;
    for (auto& read : read_)
      if (read.OverlapsWith(slice)) return true;
    return false;
  };

  // Returns true if slice overlaps with any of the slices in write set.
  auto write_overlap = [&](const BufferAllocation::Slice& slice) {
    if (write_.contains(slice)) return true;
    for (auto& write : write_)
      if (write.OverlapsWith(slice)) return true;
    return false;
  };

  return use.access() == MemoryAccess::kWrite
             ? write_overlap(use.slice()) || read_overlap(use.slice())
             : write_overlap(use.slice());
}

bool BufferUse::ReadWriteSet::HasConflicts(
    absl::Span<const BufferUse> uses) const {
  return absl::c_any_of(
      uses, [&](const BufferUse& use) { return HasConflicts(use); });
}

bool BufferUse::ReadWriteSet::HasConflicts(const ReadWriteSet& other) {
  return absl::c_any_of(other.read_,
                        [&](const BufferAllocation::Slice& slice) {
                          return HasConflicts({slice, BufferUse::kRead});
                        }) ||
         absl::c_any_of(other.write_,
                        [&](const BufferAllocation::Slice& slice) {
                          return HasConflicts({slice, BufferUse::kWrite});
                        });
}

}  // namespace xla
