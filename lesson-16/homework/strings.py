# Homework:

## 1. 100 NumPy Exercises

# This is a collection of exercises that have been collected in the NumPy mailing list, on Stack Overflow,
# and in the NumPy documentation. The goal of this collection is to offer a quick reference for both old
# and new users but also to provide a set of exercises for those who teach.

# File automatically generated. See the documentation to update questions/answers/hints programmatically.

# ## Exercises

# (Exercises 1 to 50 are listed above)

#### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)

#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)

#### 53. How to convert a float (32 bits) array into an integer (32 bits) in place? (★★☆)

#### 54. How to read the following file? (★★☆)

# 1, 2, 3, 4, 5
# 6,  ,  , 7, 8
#  ,  , 9,10,11

#### 55. What is the equivalent of enumerate for NumPy arrays? (★★☆)

#### 56. Generate a generic 2D Gaussian-like array (★★☆)

#### 57. How to randomly place p elements in a 2D array? (★★☆)

#### 58. Subtract the mean of each row of a matrix (★★☆)

#### 59. How to sort an array by the nth column? (★★☆)

#### 60. How to tell if a given 2D array has null columns? (★★☆)

#### 61. Find the nearest value from a given value in an array (★★☆)

#### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)

#### 63. Create an array class that has a name attribute (★★☆)

#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)

#### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)

#### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★☆)

#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)

#### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset indices? (★★★)

#### 69. How to get the diagonal of a dot product? (★★★)

#### 70. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)

#### 71. Consider an array of dimension (5,5,3), how to multiply it by an array with dimensions (5,5)? (★★★)

#### 72. How to swap two rows of an array? (★★★)

#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the triangles (★★★)

#### 74. Given a sorted array C that corresponds to a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)

#### 75. How to compute averages using a sliding window over an array? (★★★)

#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1])) (★★★)

#### 77. How to negate a boolean, or to change the sign of a float in place? (★★★)

#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])? (★★★)

#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])? (★★★)

#### 80. Consider an arbitrary array, write a function that extracts a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★)

#### 81. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], ..., [11,12,13,14]]? (★★★)

#### 82. Compute a matrix rank (★★★)

#### 83. How to find the most frequent value in an array? (★★★)

#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)

#### 85. Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★)

#### 86. Consider a set of p matrices with shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of the p matrix products at once? (result has shape (n,1)) (★★★)

#### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)

#### 88. How to implement the Game of Life using NumPy arrays? (★★★)

#### 89. How to get the n largest values of an array? (★★★)

#### 90. Given an arbitrary number of vectors, build the cartesian product (every combination of every item) (★★★)

#### 91. How to create a record array from a regular array? (★★★)

#### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)

#### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)

#### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) (★★★)

#### 95. Convert a vector of ints into a matrix binary representation (★★★)

#### 96. Given a two dimensional array, how to extract unique rows? (★★★)

#### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)

#### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples? (★★★)

#### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees (i.e., rows which only contain integers and which sum to n). (★★★)

#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)




## 2. 
    # - Create a NumPy array of integers from 1 to 10 (inclusive).
    # - Calculate the square of each element in the array.
    # - Find the sum, mean, and standard deviation of the squared array.

import numpy as np

# 51
dtype = np.dtype([
    ('position', np.int32, (2,)),  # position with (x, y)
    ('color', np.uint8, (3,))  # color with (r,g,b)
])

# tructured array with 3 elements
data = np.zeros(3, dtype=dtype)
data['position'] = np.random.randint(-100, 101, (3,2))
data['color'] = np.random.randint(0, 256, (3,3),dtype=np.uint8)

print(data)


import numpy as np

# 52
Z = np.random.rand(100, 2)
dist = np.linalg.norm(Z[:, None] - Z, axis=-1)

# 53
A = np.arange(10, dtype=np.float32)
A_int = A.view(np.int32)

# 54
from io import StringIO
data = "1, 2, 3, 4, 5\n6,  ,  , 7, 8\n ,  , 9,10,11"
Z = np.genfromtxt(StringIO(data), delimiter=",", dtype=float)

# 55
Z = np.arange(9).reshape(3, 3)
for idx, val in np.ndenumerate(Z):
    pass  # You can use (idx, val)

# 56
def gaussian2d(shape=100, sigma=10):
    m = np.linspace(-1, 1, shape)
    X, Y = np.meshgrid(m, m)
    D = np.sqrt(X**2 + Y**2)
    G = np.exp(-(D**2 / (2 * sigma**2)))
    return G

# 57
n, p = 10, 3
Z = np.zeros((n, n))
idx = np.random.choice(n*n, p, replace=False)
Z[np.unravel_index(idx, Z.shape)] = 1

# 58
X = np.random.rand(5, 10)
Y = X - X.mean(axis=1, keepdims=True)

# 59
A = np.random.rand(10, 3)
A_sorted = A[A[:, 1].argsort()]

# 60
Z = np.random.rand(5, 5)
has_null_cols = np.any(np.all(Z == 0, axis=0))

# 61
A = np.random.rand(10)
val = 0.5
closest = A.flat[np.abs(A - val).argmin()]

# 62
A, B = np.arange(3).reshape(1, 3), np.arange(3).reshape(3, 1)
it = np.nditer([A, B, None])
for x, y, z in it:
    z[...] = x + y
sum_res = it.operands[2]

# 63
class NamedArray(np.ndarray):
    def __new__(cls, input_array, name="default"):
        obj = np.asarray(input_array).view(cls)
        obj.name = name
        return obj

# 64
Z = np.zeros(10)
I = [1, 1, 3, 3, 5]
np.add.at(Z, I, 1)

# 65
X = [1, 2, 3, 4, 5]
I = [0, 1, 1, 3, 4]
F = np.zeros(5)
np.add.at(F, I, X)

# 66
img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
n_colors = len(np.unique(img.reshape(-1, 3), axis=0))

# 67
A = np.random.rand(3, 4, 5, 6)
sum_last2 = A.sum(axis=(-2, -1))

# 68
D = np.random.rand(100)
S = np.random.randint(0, 10, 100)
mean_by_group = np.bincount(S, D) / np.bincount(S)

# 69
A = np.random.rand(3, 3)
B = np.random.rand(3, 3)
diag_dot = np.sum(A * B.T, axis=1)

# 70
Z = np.array([1, 2, 3, 4, 5])
R = np.zeros(len(Z) * 4 - 3, dtype=int)
R[::4] = Z

# 71
A = np.random.rand(5, 5, 3)
B = np.random.rand(5, 5)
C = A * B[:, :, None]

# 72
A = np.eye(5)
A[[0, 1]] = A[[1, 0]]

# 73
T = np.random.randint(0, 100, (10, 3))
edges = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [0, 2]]]), axis=1)
unique_edges = np.unique(edges, axis=0)

# 74
C = np.array([0, 3, 1, 0])
A = np.repeat(np.arange(len(C)), C)

# 75
Z = np.arange(10)
window = 3
res = np.convolve(Z, np.ones(window) / window, mode='valid')

# 76
Z = np.arange(10)
stride = 3
out = np.lib.stride_tricks.sliding_window_view(Z, window_shape=stride)

# 77
Z = np.array([1, -2, 3])
Z *= -1

# 78
P0 = np.random.rand(10, 2)
P1 = np.random.rand(10, 2)
p = np.random.rand(2)
d = np.cross(P1 - P0, P0 - p)
l = np.linalg.norm(P1 - P0, axis=1)
dist = np.abs(d) / l

# 79
P0 = np.random.rand(5, 2)
P1 = np.random.rand(5, 2)
P = np.random.rand(10, 2)

v = P1 - P0
u = P[:, None, :] - P0
d = np.abs(np.cross(v, u)) / np.linalg.norm(v, axis=1)

# 80
def extract_subarray(A, center, shape, fill=0):
    pad_width = [(s//2, s - s//2 - 1) for s in shape]
    A_padded = np.pad(A, pad_width, mode='constant', constant_values=fill)
    center_padded = [c + p[0] for c, p in zip(center, pad_width)]
    slices = tuple(slice(c - s//2, c + s//2 + 1) for c, s in zip(center_padded, shape))
    return A_padded[slices]

# 81
Z = np.arange(1, 15)
R = np.lib.stride_tricks.sliding_window_view(Z, 4)

# 82
M = np.random.rand(10, 10)
rank = np.linalg.matrix_rank(M)

# 83
Z = np.random.randint(0, 10, 50)
most_freq = np.bincount(Z).argmax()

# 84
Z = np.random.rand(10, 10)
blocks = np.lib.stride_tricks.sliding_window_view(Z, (3, 3))

# 85
class SymmetricArray(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(idx) == 2:
            i, j = idx
            if np.isscalar(i) and np.isscalar(j):
                return super().__getitem__((max(i, j), min(i, j)))
        return super().__getitem__(idx)

Z = SymmetricArray(np.random.rand(5, 5))

# 86
p, n = 10, 4
M = np.random.rand(p, n, n)
V = np.random.rand(p, n, 1)
res = np.sum(np.matmul(M, V), axis=0)

# 87
X = np.random.rand(16, 16)
k = 4
block_sum = X.reshape(4, k, 4, k).sum(axis=(1, 3))

# 88
def game_of_life(Z, steps=1):
    for _ in range(steps):
        N = sum(np.roll(np.roll(Z, i, 0), j, 1)
                for i in (-1, 0, 1) for j in (-1, 0, 1)
                if (i, j) != (0, 0))
        Z = (N == 3) | (Z & (N == 2))
    return Z

# 89
Z = np.random.rand(100)
top_n = np.partition(Z, -5)[-5:]

# 90
import itertools
cart = np.array(list(itertools.product([1, 2], [3, 4], [5, 6])))

# 91
Z = np.array([(1, 2.0), (3, 4.0)], dtype=[('x', int), ('y', float)])

# 92
Z = np.arange(10)
res1 = Z**3
res2 = np.power(Z, 3)
res3 = np.multiply(Z, Z*Z)

# 93
A = np.random.randint(0, 5, (8, 3))
B = np.random.randint(0, 5, (2, 2))
mask = np.array([all(np.isin(b, a)) for a in A for b in B]).reshape(len(A), -1)
filtered_rows = A[mask.any(axis=1)]

# 94
Z = np.random.randint(0, 5, (10, 3))
unique_rows = Z[~np.all(Z == Z[:, [0]], axis=1)]

# 95
Z = np.array([0, 1, 2, 3, 4])
binary = ((Z[:, None] & (1 << np.arange(8))) > 0).astype(int)

# 96
Z = np.random.randint(0, 2, (6, 3))
unique = np.unique(Z, axis=0)

# 97
A = np.arange(3)
B = np.arange(3)
inner = np.einsum('i,i->', A, B)
outer = np.einsum('i,j->ij', A, B)
sum_ = np.einsum('i->', A)
mul = np.einsum('i->i', A * B)

# 98
X = np.cumsum(np.random.rand(10))
Y = np.cumsum(np.random.rand(10))
distances = np.sqrt(np.diff(X)**2 + np.diff(Y)**2)
dist = np.insert(np.cumsum(distances), 0, 0)
from scipy import interpolate
f = interpolate.interp1d(dist, np.c_[X, Y], axis=0)
equidistant = f(np.linspace(0, dist[-1], 50))

# 99
n = 10
X = np.random.randint(0, 5, (100, 5))
valid = X.sum(axis=1) == n
valid &= np.all(X == X.astype(int), axis=1)
result = X[valid]

# 100
X = np.random.rand(1000)
N = 1000
means = np.random.choice(X, (N, len(X)), replace=True).mean(axis=1)
conf_int = np.percentile(means, [2.5, 97.5])



## 2. 
arr = np.arange(1, 11)
squared_arr = arr **2
print(squared_arr)
print(squared_arr.sum())
print(squared_arr.mean())
print(squared_arr.std())