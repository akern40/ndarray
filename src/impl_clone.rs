// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::imp_prelude::*;
use crate::ArrayPartsSized;
use crate::RawDataClone;

impl<S: RawDataClone, D: Clone> Clone for ArrayBase<S, StridedLayout<D>>
{
    fn clone(&self) -> ArrayBase<S, StridedLayout<D>>
    {
        // safe because `clone_with_ptr` promises to provide equivalent data and ptr
        unsafe {
            let (data, ptr) = self.data.clone_with_ptr(self.parts.handle);
            ArrayBase {
                data,
                parts: ArrayPartsSized::new(
                    ptr,
                    StridedLayout::new(self.parts.layout.dim.clone(), self.parts.layout.strides.clone()),
                ),
            }
        }
    }

    /// `Array` implements `.clone_from()` to reuse an array's existing
    /// allocation. Semantically equivalent to `*self = other.clone()`, but
    /// potentially more efficient.
    fn clone_from(&mut self, other: &Self)
    {
        unsafe {
            self.parts.handle = self
                .data
                .clone_from_with_ptr(&other.data, other.parts.handle);
            self.parts.layout.dim.clone_from(&other.parts.layout.dim);
            self.parts
                .layout
                .strides
                .clone_from(&other.parts.layout.strides);
        }
    }
}

impl<S: RawDataClone + Copy, D: Copy> Copy for ArrayBase<S, StridedLayout<D>> {}
