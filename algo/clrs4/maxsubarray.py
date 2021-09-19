"""
CLRS Chapter 4.1

Algorithms for the maximum subarray problem.

Three implementations:
    1) Naive: try all pairs of buy/sell dates to find the best

    2) Divide and conquer:
        - divide array A into left A[1..mid] and right A[mid+1..N]
        - recursively find the max subarrays of left and right halves
        - find the maximum "crossing" subarray of the form A[i..mid..j]
        - the answer is the maximum of {left, crossing, right}

    3) Dynamic programming: using the simple optimal substructure of this problem

Some things to note:
    - Open this in vscode to get Jupyter-like bells and whistles

    - dnc = Divide aNd Conquer

    - The index calculations get a bit confusing (for me, atleast :/)
        - _naive and _dnc functions are user-facing: these would be part of an API if you were bundling them in a library or something

        - internal functions such as suffixed by _recurse, _cross are non-user-facing

        - user facing functions are supposed to return index ranges right-exclusive; i.e., (i,j) tuple corresponds to slice i:j = i, i+1 , i+2 , .. , j-1

        - non-user-facing functions return index ranges that are right-inclusive, i.e., (i,j) means i, i+1, i+2, .. , j-1, j
"""


# %%
def maxsubarray_naive(A):
    """
    Finds the max subarray given an array A. This is the naive O(n^2) version.

    Params:
        A -> the array to search
    Returns:
        A tuple (sum, (i,j)) where sum = the max sum found and (i,j) = python style indices range corresponding to the max subarray found, i.e., A[i:j]
    """
    if not A:
        return [0, (-1,-1)]

    globalBest = None
    best_i, best_j = None, None
    for i in range(len(A)):  # pick a buy date
        subarraySum = 0  # sum of A[i..j]
        for j in range(i,len(A)):  # pick a sell date
            subarraySum += A[j]
            if globalBest is None or globalBest < subarraySum:
                globalBest = subarraySum
                best_i, best_j = i, j

    return (globalBest, (best_i, best_j+1))


# %%
def maxsubarray_dnc(A):
    """
    Divide and conquer the maximum subarray problem given an array. This is the
    O(nlog(n)) version.

    Params:
        A -> the input array of numbers

    Returns:
        the maximum subarray sum and indices in a tuple (sum, (i,j))
        where A[i:j] is the subarray found
    """
    maxSum, (i,j) = maxsubarray_recurse(A, 0, len(A)-1)
    return maxSum, (i,j+1)

n0 = 40
def maxsubarray_recurse(A, low, high):
    """
    Recursively compute maximum subarray given list A[low..high]

    Params:
        A -> the list
        low -> the start index
        high -> the end index (inclusive)

    Returns:
        The maximum subarray in a tuple (sum, (i,j))

        NOTE: (i,j) returned is inclusive of both ends
    """
    # base case: singleton list
    # if low == high:
    #     return A[low], (low, high)

    # change base case to apply naive algorithm when input size < n0
    # n0 found below by plotting the runtime
    global n0
    if high - low + 1 < n0:
        return maxsubarray_naive(A)

    mid = low + (high-low)//2
    maxSumLeft, leftIndices = maxsubarray_recurse(A, low, mid)
    maxSumRight, rightIndices = maxsubarray_recurse(A, mid+1, high)
    maxSumCross, crossIndices = maxsubarray_cross(A, low, mid, high)

    # Either left, right or crossing wins
    if maxSumLeft >= maxSumRight and maxSumLeft >= maxSumCross:
        return maxSumLeft, leftIndices
    elif maxSumRight >= maxSumLeft and maxSumRight >= maxSumCross:
        return maxSumRight, rightIndices
    else:
        return maxSumCross, crossIndices


def maxsubarray_cross(A, low, mid, high):
    """
    Finds the maximum crossing subarray of the form A[i..mid..j]
    where low <= i <= mid < j <= high

    Params:
        A -> array
        low -> lower limit
        high -> upper limit

    Returns:
        The maximum crossing subarray (i.e., passing through mid) as a tuple
        (sum, (i,j))
    """
    currentSum = 0

    # Invariants:
    #   bestLeft is the best subarray sum in A[i..mid]
    #   bestLeftIndex is the latest index k in i..mid that produced bestLeft
    #   currentSum is the sum of entries A[i..mid]
    bestLeft = None
    bestLeftIndex = mid
    for i in range(mid, low-1, -1):
        currentSum += A[i]
        if bestLeft is None or bestLeft < currentSum:
            bestLeft = currentSum
            bestLeftIndex = i

    currentSum = 0  # reset the sum

    # Invariants:
    #   bestRight is the best subarray sum in A[mid+1..j]
    #   bestRightIndex is the latest index k in mid+1..j that produced bestRight
    #   currentSum is the sum of entries A[mid+1..j]
    bestRight = None
    bestRightIndex = mid
    for j in range(mid+1, high+1):
        currentSum += A[j]
        if bestRight is None or bestRight < currentSum:
            bestRight = currentSum
            bestRightIndex = j

    return (
        bestLeft+bestRight
        , (bestLeftIndex, bestRightIndex)
        )


#%%
import random
import time

SEED = 0xCAFE42
random.seed(SEED)

SAMPLE_INPUTS = []  # list of randomly generated inputs with some bookkeeping
NUM_INPUTS = 50  # number of inputs to generate
INPUT_LEN_RANGE = (0,10)  # the possible lengths of each input
INPUT_VAL_RANGE = (-100,100)  # the range of values within each input

# generate some random inputs
for _ in range(NUM_INPUTS):
    N = random.randint(*INPUT_LEN_RANGE)

    SAMPLE_INPUTS.append({
        'input': [random.randint(*INPUT_VAL_RANGE) for _ in range(N)]
        , 'solution': {'maxSum': None, 'indexRange': None}
    })


#%%
# generate and store known-to-be-correct answers in the bookkeeping section
# better than eyeballing ;)
for sample_input in SAMPLE_INPUTS:
    bestSum, indices = maxsubarray_naive(sample_input['input'])
    sample_input['solution']['maxSum'] = bestSum
    sample_input['solution']['indexRange'] = indices

#%% Timing functions
class TimedFunction:
    """
    A very stupid context manager that could have been a much shorter decorator.
    Too lazy to change it now; perhaps one day I will.
    """
    def __init__(self, f, iters=5):
        self.f = f
        self.logs = []
        self.iters = iters

    def __enter__(self):
        return self

    def __exit__(self, *pargs):
        pass

    def __call__(self, *pargs, **kwargs):
        runs = []
        for _ in range(self.iters):
            start = time.perf_counter_ns()
            result = self.f(*pargs, **kwargs)
            runs.append(time.perf_counter_ns() - start)

        self.logs.append(sum(runs)/len(runs))
        return result


# %%
import matplotlib.pyplot as plt

maxsub_naive = TimedFunction(maxsubarray_naive)
maxsub_dnc = TimedFunction(maxsubarray_dnc)
inputsizes = [n for n in range(1,100+1)]

with maxsub_naive as maxsub:
    for n in inputsizes:
        maxsub([
            random.randint(*INPUT_VAL_RANGE)
            for _ in range(n)
        ])

with maxsub_dnc as maxsub:
    for n in inputsizes:
        maxsub([
            random.randint(*INPUT_VAL_RANGE)
            for _ in range(n)
        ])

axes = plt.gca()
axes.scatter(inputsizes, maxsub_naive.logs)
axes.scatter(inputsizes, maxsub_dnc.logs)
axes.set_xticks([n for n in inputsizes if n%5 == 0])
n0 = 40  # crossover point where _dnc starts to beat _naive
n0_next = 50  # new crossover point when base case is solved by _naive
# axes.axvline(x=n0)

# %%
def maxsubarray_dp(A):
    """
    An O(n) algorithm (n = len(A)) for the maximum subarray problem using the dynamic programming formulation that observes:
        a maximum subarray of A[1..N] is either
            1) A maximum subarray of A[1..N-1] (i.e., does not include A[N])
            2) A maximum subarray of the form A[i..N] with 1 <= i <= N (i.e., does include A[N])

    Essentially, at each point as we scan an array left-to-right, the maximum of (1) and (2) so far corresponds to the maximum subarray so far.

    More formally, let:
        M(j) = maximum subarray sum of A[1..j]
        E(j) = maximum subarray sum ending at, and including, A[j]

    So now observe that:
        M(N) = max( M(N-1), E(N) )
        i.e., maximum subarray of A[1..N] is either the maximum found in A[1..N-1] or the maximum ending at index N, whichever is bigger

        E(N) = max( A[N], E(N-1) + A[N] )
        i.e., the maximum ending at index N is either A[N] by itself, or A[N] + maximum ending at index (N-1), whichever is bigger

        Now we just need to notice that if max(A[N], E(N-1)+A[N]) = A[N] then we must have that E(N-1) < 0, which is the test we use to find the maximum ending at N.

        It is instructive to try out the recurrence by hand on a few examples first. My puny brain took a while to see the pattern. An algorithm must really be seen to be believed.

    Params:
        A -> the array

    Returns:
        (sum, (i,j)) where
            sum = the subarray sum
            (i,j) = the index range i,i+1,..,j-1
    """
    m = 0  # maximum found so far in A[:q+1]
    i, j = 0, 0  # i,j are the indices corresponding to max so far

    p = 0  # A[p:q+1] is maximum subarray ending at index q
    mq = 0  # mq is the sum of A[p:q+1]
    for q in range(len(A)):
        if mq < 0:
            p = q
            mq = A[q]
        else:
            mq += A[q]

        if m < mq:
            m = mq
            i, j = p, q

    return (m, (i,j+1))
# %%
