#!/usr/bin/env python3
"""Debug data layout issues."""

import struct
import numpy as np

def read_raw(path):
    with open(path, 'rb') as f:
        shape = [struct.unpack('<q', f.read(8))[0] for _ in range(4)]
        data = np.frombuffer(f.read(), dtype=np.float32)
        return shape, data

# Read manual_enc_layer0_out.bin
shape, data = read_raw("test_data/reference/manual_enc_layer0_out.bin")
print(f"manual_enc_layer0_out:")
print(f"  GGML shape (reversed): {shape}")
print(f"  Data length: {len(data)}")
print(f"  First 10 values: {data[:10]}")
print(f"  Values at indices 925, 926: {data[925]}, {data[926]}")

# Interpret shape
# GGML reversed order means if stored [a, b, c, d], original was [d, c, b, a]
# So [1, 768, 14, 1] means PyTorch was [1, 14, 768] (ignoring trailing 1s)
print(f"\n  Interpreted PyTorch shape: [1, 14, 768]")
print(f"  Total elements: 1 * 14 * 768 = {1*14*768}")

# In PyTorch [1, 14, 768] row-major:
# data[b, t, d] = data[(b * 14 + t) * 768 + d]
# So data[0, 1, 157] = data[(0*14 + 1) * 768 + 157] = data[768 + 157] = data[925]
print(f"\n  data[0, 1, 157] (Python row-major) = data[{768+157}] = {data[925]}")

# In GGML [768, 14] column-major:
# data[d, t] = data[d + t * 768]
# So data[157, 1] = data[157 + 1 * 768] = data[925]
# This is the same!
print(f"  data[157, 1] (GGML col-major) = data[157 + 1*768] = {data[157 + 768]}")

# So position 925 in both layouts refers to the same element: token 1, dim 157
