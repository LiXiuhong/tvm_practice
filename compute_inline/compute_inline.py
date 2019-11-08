# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
.. _opt-conv-gpu:

How to optimize convolution on GPU
==================================
**Author**: `Haichen Shen <https://homes.cs.washington.edu/~haichen/>`_

In this tutorial, we will demonstrate how to write a high performance
convolution implementation in TVM. We use square size input tensors and filters
as an example, and assume the input to convolution has a large batch. In this
example, we use a different layout to store the data in order to achieve better
data locality. The buffer layout is HWCN, which stands for height, width,
channel, batch.

"""

################################################################
# Preparation and Algorithm
# -------------------------
#
# We use the fixed size for input tensors with 256 channels and 14 x 14
# dimensions. The batch size is 256. Convolution filters contain 512 filters
# of size 3 x 3.  We use stride size 1 and padding size 1 for the
# convolution. The following code defines the convolution algorithm in TVM.
#

import numpy as np
import tvm

# The sizes of inputs and filters
batch = 256
in_channel = 256
out_channel = 512
in_size = 14
kernel = 3
pad = 1
stride = 1

# Algorithm
A = tvm.placeholder((in_size, in_size, in_channel, batch), name='A')
W = tvm.placeholder((kernel, kernel, in_channel, out_channel), name='W')
out_size = (in_size - kernel + 2*pad) // stride + 1
# Pad input
Apad = tvm.compute(
    (in_size + 2*pad, in_size + 2*pad, in_channel, batch),
    lambda yy, xx, cc, nn: tvm.if_then_else(
        tvm.all(yy >= pad, yy - pad < in_size,
                xx >= pad, xx - pad < in_size),
        A[yy - pad, xx - pad, cc, nn], tvm.const(0., "float32")),
    name='Apad')
# Create reduction variables
rc = tvm.reduce_axis((0, in_channel), name='rc')
ry = tvm.reduce_axis((0, kernel), name='ry')
rx = tvm.reduce_axis((0, kernel), name='rx')
# Compute the convolution
B = tvm.compute(
    (out_size, out_size, out_channel, batch),
    lambda yy, xx, ff, nn: tvm.sum(
        Apad[yy * stride + ry, xx * stride + rx, rc, nn] * W[ry, rx, rc, ff],
        axis=[ry, rx, rc]),
    name='B')


s = tvm.create_schedule(B.op)
s[Apad].compute_inline() # compute Apad inline
func = tvm.build(s, [A, W, B], 'c')
print("/*")
print(tvm.lower(s, [A, W, B], simple_mode=True))
print("*/")
