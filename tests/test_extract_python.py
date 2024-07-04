import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.textprocessing import extract_python_code
import traceback



tc_1_in = """
Certainly, Here is a simple function that adds two numbers:
def add(x, y):
    return x + y

This function simply adds two numbers.
"""

tc_1_out = """
def add(x, y):
    return x + y
"""

tc_2_in = """
of course, here is a nested function:

    it is interesting if I indent here
    
and dedent 

import os

def outer_function(a):
    # This function wraps another function
    def inner_function(b):
        return b * 2
    return inner_function(a)

# End of nested functions

"""

tc_2_out = """
import os
def outer_function(a):
    # This function wraps another function
    def inner_function(b):
        return b * 2
    return inner_function(a)

# End of nested functions
"""

tc_3_in = """
of course here are some imports:
import os
import sys
from datetime import datetime
import re as regex
if __name__ == "__main__":
    # This is the main block
    print("This is a test.")
    for i in range(5):
        print(i)
# Main block ends above
"""

tc_3_out = """
import os
import sys
from datetime import datetime
import re as regex
if __name__ == "__main__":
    # This is the main block
    print("This is a test.")
    for i in range(5):
        print(i)
# Main block ends above
"""

tc_4_in = """
def incorrect_indent():
print("This should not work")
"""

tc_4_out = "def incorrect_indent():"


tc_5_in = """
import os, sys, \
    datetime, re

def wrapped_line_function():
    x = 1 + 2 + \
        3 + 4
    return x
"""

tc_5_out = """
import os, sys, \
    datetime, re
def wrapped_line_function():
    x = 1 + 2 + \
        3 + 4
    return x
"""

tc_6_in = """
here is a function: 
import foobar
def f(x):
    import inner as i
    from outer import outer
    return x
we did it fam
import baloney
"""

tc_6_out = """
import foobar
def f(x):
    import inner as i
    from outer import outer
    return x

import baloney
"""

tc_7_in = """
this is some text
and more
class A:
    def __init__(self):
        print("init")
    def f(self):
        print("f")
dont extract this
def g():
    print("g")
"""

tc_7_out = """
class A:
    def __init__(self):
        print("init")
    def f(self):
        print("f")

def g():
    print("g")
"""

tc_8_in = """
@decorator
class A:
    def __init__(self):
        print("init")
    def f(self):
        print("f")
dont extract this
"""

tc_8_out = """
@decorator
class A:
    def __init__(self):
        print("init")
    def f(self):
        print("f")
"""

tc_9_in = """
foobar 
class Outer:
    class Inner:
        def __init__(self):
            print("init")
        def f(self):
            print("f")

    def g(self):
        print("g")

barfoo
"""

tc_9_out = """
class Outer:
    class Inner:
        def __init__(self):
            print("init")
        def f(self):
            print("f")

    def g(self):
        print("g")
"""

tc_10_in = """
Your python code using the concept of Segment Tree to solve this problem:

```Python
import sys
from typing import List, Tuple

class SegmentTree:
    def __init__(self, n):
        self.sum_tree = [0] * (4 * n)
        self.range_tree = [0] * (4 * n)
        self.size = 4 * n

    def build_tree(self, arr, node, start, end):
        if start == end:
            self.sum_tree[node] = arr[start]
            self.range_tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self.build_tree(arr, 2 * node, start, mid)
            self.build_tree(arr, 2 * node + 1, mid + 1, end)
            self.sum_tree[node] = self.sum_tree[2 * node] + self.sum_tree[2 * node + 1]
            self.range_tree[node] = max(self.range_tree[2 * node], self.range_tree[2 * node + 1])

    def update_tree(self, node, start, end, x, y, val):
        if start == end:
            self.sum_tree[node] -= val
            self.range_tree[node] = max(0, self.range_tree[node] - val)
        else:
            mid = (start + end) // 2
            if x <= mid:
                self.update_tree(2 * node, start, mid, x, y, val)
            elif x > mid:
                self.update_tree(2 * node + 1, mid + 1, end, x, y, val)
            else:
                self.update_tree(2 * node, start, mid, x, y, val)
                self.update_tree(2 * node + 1, mid + 1, end, x, y, val)
            self.sum_tree[node] = self.sum_tree[2 * node] + self.sum_tree[2 * node + 1]
            self.range_tree[node] = max(self.range_tree[2 * node], self.range_tree[2 * node + 1])

    def query_tree(self, node, start, end, l, r):
        if start > r or end < l:
            return 0
        if start >= l and end <= r:
            return self.range_tree[node]
        mid = (start + end) // 2
        p1 = self.query_tree(2 * node, start, mid, l, r)
        p2 = self.query_tree(2 * node + 1, mid + 1, end, l, r)
        return max(p1, p2)

def f(N: int, M: int, triples: List[Tuple[int, int, int]], Q: int, pairs: List[Tuple[int, int]]) -> None:
    st = SegmentTree(N)
    st.build_tree([0] * N, 1, 0, N - 1)
    for a, b, c in triples:
        st.update_tree(1, 0, N - 1, a, min(b, N - 1), -c)
    for d, e in pairs:
        res = st.query_tree(1, 0, N - 1, d, min(e, N - 1))
        print(res)
"""


tc_10_out = """
import sys
from typing import List, Tuple
class SegmentTree:
    def __init__(self, n):
        self.sum_tree = [0] * (4 * n)
        self.range_tree = [0] * (4 * n)
        self.size = 4 * n

    def build_tree(self, arr, node, start, end):
        if start == end:
            self.sum_tree[node] = arr[start]
            self.range_tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self.build_tree(arr, 2 * node, start, mid)
            self.build_tree(arr, 2 * node + 1, mid + 1, end)
            self.sum_tree[node] = self.sum_tree[2 * node] + self.sum_tree[2 * node + 1]
            self.range_tree[node] = max(self.range_tree[2 * node], self.range_tree[2 * node + 1])

    def update_tree(self, node, start, end, x, y, val):
        if start == end:
            self.sum_tree[node] -= val
            self.range_tree[node] = max(0, self.range_tree[node] - val)
        else:
            mid = (start + end) // 2
            if x <= mid:
                self.update_tree(2 * node, start, mid, x, y, val)
            elif x > mid:
                self.update_tree(2 * node + 1, mid + 1, end, x, y, val)
            else:
                self.update_tree(2 * node, start, mid, x, y, val)
                self.update_tree(2 * node + 1, mid + 1, end, x, y, val)
            self.sum_tree[node] = self.sum_tree[2 * node] + self.sum_tree[2 * node + 1]
            self.range_tree[node] = max(self.range_tree[2 * node], self.range_tree[2 * node + 1])

    def query_tree(self, node, start, end, l, r):
        if start > r or end < l:
            return 0
        if start >= l and end <= r:
            return self.range_tree[node]
        mid = (start + end) // 2
        p1 = self.query_tree(2 * node, start, mid, l, r)
        p2 = self.query_tree(2 * node + 1, mid + 1, end, l, r)
        return max(p1, p2)

def f(N: int, M: int, triples: List[Tuple[int, int, int]], Q: int, pairs: List[Tuple[int, int]]) -> None:
    st = SegmentTree(N)
    st.build_tree([0] * N, 1, 0, N - 1)
    for a, b, c in triples:
        st.update_tree(1, 0, N - 1, a, min(b, N - 1), -c)
    for d, e in pairs:
        res = st.query_tree(1, 0, N - 1, d, min(e, N - 1))
        print(res)
"""


tc_11_in = """
Your python code using the concept of Segment Tree to solve this problem:

```Python
    import sys
    from typing import List, Tuple

    class SegmentTree:
        def __init__(self, n):
            self.sum_tree = [0] * (4 * n)
            self.range_tree = [0] * (4 * n)
            self.size = 4 * n

        def build_tree(self, arr, node, start, end):
            if start == end:
                self.sum_tree[node] = arr[start]
                self.range_tree[node] = arr[start]
            else:
                mid = (start + end) // 2
                self.build_tree(arr, 2 * node, start, mid)
                self.build_tree(arr, 2 * node + 1, mid + 1, end)
                self.sum_tree[node] = self.sum_tree[2 * node] + self.sum_tree[2 * node + 1]
                self.range_tree[node] = max(self.range_tree[2 * node], self.range_tree[2 * node + 1])

        def update_tree(self, node, start, end, x, y, val):
            if start == end:
                self.sum_tree[node] -= val
                self.range_tree[node] = max(0, self.range_tree[node] - val)
            else:
                mid = (start + end) // 2
                if x <= mid:
                    self.update_tree(2 * node, start, mid, x, y, val)
                elif x > mid:
                    self.update_tree(2 * node + 1, mid + 1, end, x, y, val)
                else:
                    self.update_tree(2 * node, start, mid, x, y, val)
                    self.update_tree(2 * node + 1, mid + 1, end, x, y, val)
                self.sum_tree[node] = self.sum_tree[2 * node] + self.sum_tree[2 * node + 1]
                self.range_tree[node] = max(self.range_tree[2 * node], self.range_tree[2 * node + 1])

        def query_tree(self, node, start, end, l, r):
            if start > r or end < l:
                return 0
            if start >= l and end <= r:
                return self.range_tree[node]
            mid = (start + end) // 2
            p1 = self.query_tree(2 * node, start, mid, l, r)
            p2 = self.query_tree(2 * node + 1, mid + 1, end, l, r)
            return max(p1, p2)

    def f(N: int, M: int, triples: List[Tuple[int, int, int]], Q: int, pairs: List[Tuple[int, int]]) -> None:
        st = SegmentTree(N)
        st.build_tree([0] * N, 1, 0, N - 1)
        for a, b, c in triples:
            st.update_tree(1, 0, N - 1, a, min(b, N - 1), -c)
        for d, e in pairs:
            res = st.query_tree(1, 0, N - 1, d, min(e, N - 1))
            print(res)
"""


tc_11_out = """
import sys
from typing import List, Tuple
class SegmentTree:
    def __init__(self, n):
        self.sum_tree = [0] * (4 * n)
        self.range_tree = [0] * (4 * n)
        self.size = 4 * n

    def build_tree(self, arr, node, start, end):
        if start == end:
            self.sum_tree[node] = arr[start]
            self.range_tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self.build_tree(arr, 2 * node, start, mid)
            self.build_tree(arr, 2 * node + 1, mid + 1, end)
            self.sum_tree[node] = self.sum_tree[2 * node] + self.sum_tree[2 * node + 1]
            self.range_tree[node] = max(self.range_tree[2 * node], self.range_tree[2 * node + 1])

    def update_tree(self, node, start, end, x, y, val):
        if start == end:
            self.sum_tree[node] -= val
            self.range_tree[node] = max(0, self.range_tree[node] - val)
        else:
            mid = (start + end) // 2
            if x <= mid:
                self.update_tree(2 * node, start, mid, x, y, val)
            elif x > mid:
                self.update_tree(2 * node + 1, mid + 1, end, x, y, val)
            else:
                self.update_tree(2 * node, start, mid, x, y, val)
                self.update_tree(2 * node + 1, mid + 1, end, x, y, val)
            self.sum_tree[node] = self.sum_tree[2 * node] + self.sum_tree[2 * node + 1]
            self.range_tree[node] = max(self.range_tree[2 * node], self.range_tree[2 * node + 1])

    def query_tree(self, node, start, end, l, r):
        if start > r or end < l:
            return 0
        if start >= l and end <= r:
            return self.range_tree[node]
        mid = (start + end) // 2
        p1 = self.query_tree(2 * node, start, mid, l, r)
        p2 = self.query_tree(2 * node + 1, mid + 1, end, l, r)
        return max(p1, p2)

def f(N: int, M: int, triples: List[Tuple[int, int, int]], Q: int, pairs: List[Tuple[int, int]]) -> None:
    st = SegmentTree(N)
    st.build_tree([0] * N, 1, 0, N - 1)
    for a, b, c in triples:
        st.update_tree(1, 0, N - 1, a, min(b, N - 1), -c)
    for d, e in pairs:
        res = st.query_tree(1, 0, N - 1, d, min(e, N - 1))
        print(res)
"""

tc_12_in = """

```while True:
    print("Infinite loop")
```
then a bunch of text

```Python
def f(x):
    return x + 1
```

finally some more text

"""

tc_12_out = """
while True:
    print("Infinite loop")

def f(x):
    return x + 1
"""

tc_13_in = """
here is some text
for i in range(10):
    print(i)
this
"""

tc_13_out = """
for i in range(10):
    print(i)
"""

tc_14_in = """
try to understand the : following 
if you think about the following: when you may have a colon after if, or things like that like
while True: but followed with text, 
it shouldn't match"""

tc_14_out = ""

tc_15_in = """
if 2+2 == 4:
    print("math is correct")
    
more text

elif 2+2 == 5:
    print("math is wrong")

else:
    print("math is unknown")
"""

tc_15_out = """
if 2+2 == 4:
    print("math is correct")

elif 2+2 == 5:
    print("math is wrong")

else:
    print("math is unknown")
"""

tc_16_in = """
Not this
if I have some stuff here that doesn't make sense:
    this should still pass
"""

tc_16_out = """
if I have some stuff here that doesn't make sense:
    this should still pass
"""

tc_17_in = """
#python #algorithm #math #competitive-programming 

Here is the python code to solve the problem:

```python
MOD = 10**9 + 7

def solve(X, Y):
    max_steps = 0
    max_count = 0
    for a in range(1, X + 1):
        for b in range(1, Y + 1):
            x, y = a, b
            steps = 0
            while y > 0:
                x, y = y, x % y
                steps += 1
            if steps > max_steps:
                max_steps = steps
                max_count = 1
            elif steps == max_steps:
                max_count += 1
    print(max_steps, max_count % MOD)

Q = int(input())
for _ in range(Q):
    X, Y = map(int, input().split())
    solve(X, Y)
```

This solution is quite straightforward. It simply calculates the Euclidean algorithm step count for each pair of numbers in the given range, keeps track of the maximum step count and the number of pairs that achieve it, and then prints the result.

However, this solution is very slow because it has to calculate the Euclidean algorithm step count for each pair of numbers, which can take up to O(X*Y) time. Since X and Y can be as large as 10^18, this solution will definitely timeout.

To optimize the solution, we can use dynamic programming to store the Euclidean algorithm step count for each pair of numbers. This can reduce the time complexity to O(X*Y) for the entire input, but it still requires a lot of space to store the dynamic programming table.

"""

tc_17_out = """
#python #algorithm #math #competitive-programming 
MOD = 10**9 + 7
def solve(X, Y):
    max_steps = 0
    max_count = 0
    for a in range(1, X + 1):
        for b in range(1, Y + 1):
            x, y = a, b
            steps = 0
            while y > 0:
                x, y = y, x % y
                steps += 1
            if steps > max_steps:
                max_steps = steps
                max_count = 1
            elif steps == max_steps:
                max_count += 1
    print(max_steps, max_count % MOD)

Q = int(input())
for _ in range(Q):
    X, Y = map(int, input().split())
    solve(X, Y)
"""

tc_18_in = """
import heapq
from collections import deque
from enum import Enum
import sys
import math
from _heapq import heappush, heappop
import copy
from test.support import _MemoryWatchdog
BIG_NUM = 2000000000
HUGE_NUM = 99999999999999999
MOD = 1000000007
EPS = 0.000000001
sys.setrecursionlimit(100000)
SIZE = 105
table = [0]*SIZE
N = int(input())
input_array = list(map(int,input().split()))
maximum = 0

for i in range(N):
    table[0] -= 1
    table[input_array[i]] += 1
    maximum = max(maximum,input_array[i])

for i in range(maximum-1,-1,-1):
    table[i] += table[i+1]

ans = 0

for i in range(1,maximum+1):
    if i <= table[i]:
        ans = i

print("%d"%(ans))"""

tc_18_out = """
import heapq
from collections import deque
from enum import Enum
import sys
import math
from _heapq import heappush, heappop
import copy
from test.support import _MemoryWatchdog
BIG_NUM = 2000000000
HUGE_NUM = 99999999999999999
MOD = 1000000007
EPS = 0.000000001
sys.setrecursionlimit(100000)
SIZE = 105
table = [0]*SIZE
N = int(input())
input_array = list(map(int,input().split()))
maximum = 0
for i in range(N):
    table[0] -= 1
    table[input_array[i]] += 1
    maximum = max(maximum,input_array[i])

for i in range(maximum-1,-1,-1):
    table[i] += table[i+1]

ans = 0
for i in range(1,maximum+1):
    if i <= table[i]:
        ans = i

print("%d"%(ans))"""


tc_19_in = """
a = input()
columns = a.split(' ')          #??\??????????????????????????§??????
num_of_student = int(columns[0])   #???????????°
Baton = int(columns[1])            #??????????????°

student_bib = []
for i in range(num_of_student):
    student_bib.append(int(input()))   #???????????????????????\???

for Baton_num in range(Baton + 1):   #??????????????????
    if Baton_num == 0:          #????????????????????????
        continue
    m = 0
    for j in range(num_of_student):     #???????????°?????????????????????
        if num_of_student - 1 == m:
            break
        if (student_bib[m] % Baton_num) > (student_bib[m + 1] % Baton_num):
            student_bib_tmp = student_bib[m]
            student_bib[m] = student_bib[m + 1]
            student_bib[m + 1] = student_bib_tmp  #??????????????????
        m += 1


for student_bib_ in student_bib:
    print(student_bib_)
"""

tc_19_out = """
a = input()
columns = a.split(' ')          #??\??????????????????????????§??????
num_of_student = int(columns[0])   #???????????°
Baton = int(columns[1])            #??????????????°

student_bib = []
for i in range(num_of_student):
    student_bib.append(int(input()))   #???????????????????????\???

for Baton_num in range(Baton + 1):   #??????????????????
    if Baton_num == 0:          #????????????????????????
        continue
    m = 0
    for j in range(num_of_student):     #???????????°?????????????????????
        if num_of_student - 1 == m:
            break
        if (student_bib[m] % Baton_num) > (student_bib[m + 1] % Baton_num):
            student_bib_tmp = student_bib[m]
            student_bib[m] = student_bib[m + 1]
            student_bib[m + 1] = student_bib_tmp  #??????????????????
        m += 1


for student_bib_ in student_bib:
    print(student_bib_)
"""


tc_20_in = """
import os
import sys

if os.getenv("LOCAL"):
    sys.stdin = open("_in.txt", "r")

sys.setrecursionlimit(2147483647)
INF = float("inf")
IINF = 10 ** 18
MOD = 10 ** 9 + 7

# 箱根


N = int(sys.stdin.readline())
C = [sys.stdin.readline().rstrip() for _ in range(N)]
C = [c for c in C if c != '-']
N = len(C)

# dp[i][j][k]:
# iまで見て、
# j個未割当の人が残ってて、
# j個未割当の場所が残ってる場合の数
dp = [[0] * (N + 1) for _ in range(N + 1)]
dp[0][0] = 1
for i, c in enumerate(C):
    for j in range(N):
        if c == 'D':
            if j - 1 >= 0:
                # 前回i位だった人を今まで見た場所に割り当てる
                dp[i + 1][j - 1] += dp[i][j] * j * j
                dp[i + 1][j - 1] %= MOD
            # 割り当てない
            dp[i + 1][j] += dp[i][j] * j
            dp[i + 1][j] %= MOD
        if c == 'U':
            # 前回i位だった人を今まで見た場所に割り当てる
            dp[i + 1][j] += dp[i][j] * j
            dp[i + 1][j] %= MOD
            # 割り当てない
            dp[i + 1][j + 1] += dp[i][j]
            dp[i + 1][j + 1] %= MOD
print(dp[-1][0])
"""

tc_20_out = """
import os
import sys

if os.getenv("LOCAL"):
    sys.stdin = open("_in.txt", "r")

sys.setrecursionlimit(2147483647)
INF = float("inf")
IINF = 10 ** 18
MOD = 10 ** 9 + 7

# 箱根


N = int(sys.stdin.readline())
C = [sys.stdin.readline().rstrip() for _ in range(N)]
C = [c for c in C if c != '-']
N = len(C)

# dp[i][j][k]:
# iまで見て、
# j個未割当の人が残ってて、
# j個未割当の場所が残ってる場合の数
dp = [[0] * (N + 1) for _ in range(N + 1)]
dp[0][0] = 1
for i, c in enumerate(C):
    for j in range(N):
        if c == 'D':
            if j - 1 >= 0:
                # 前回i位だった人を今まで見た場所に割り当てる
                dp[i + 1][j - 1] += dp[i][j] * j * j
                dp[i + 1][j - 1] %= MOD
            # 割り当てない
            dp[i + 1][j] += dp[i][j] * j
            dp[i + 1][j] %= MOD
        if c == 'U':
            # 前回i位だった人を今まで見た場所に割り当てる
            dp[i + 1][j] += dp[i][j] * j
            dp[i + 1][j] %= MOD
            # 割り当てない
            dp[i + 1][j + 1] += dp[i][j]
            dp[i + 1][j + 1] %= MOD
print(dp[-1][0])
"""

tc_21_in = '''
"""
Set - Multi-Set
http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=ITP2_7_D&lang=jp

"""
from bisect import insort, bisect_right, bisect_left

class Multi_set:
    def __init__(self):
        self.total = 0
        self.ms = dict()
        self.lr = []

    def insert(self, x):
        self.total += 1
        if x in self.ms:
            self.ms[x] += 1
        else:
            self.ms[x] = 1
            insort(self.lr, x)
        print(self.total)

    def find(self, x):
        print(self.ms.get(x, 0))

    def delete(self, x):
        if x in self.ms:
            self.total -= self.ms[x]
            self.ms[x] = 0

    def dump(self, l, r):
        lb = bisect_left(self.lr, l)
        ub = bisect_right(self.lr, r)
        for i in range(lb, ub):
            k = self.lr[i]
            v = self.ms[k]
            print(f'{k}\\n' * v, end='')


ms = Multi_set()
for _ in range(int(input())):
    op, x, y = (input() + ' 1').split()[:3]
    if op == '0':
        ms.insert(int(x))
    elif op == '1':
        ms.find(int(x))
    elif op == '2':
        ms.delete(int(x))
    else:
        ms.dump(int(x), int(y))

'''
        
tc_21_out = '''
"""
Set - Multi-Set
http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=ITP2_7_D&lang=jp

"""
from bisect import insort, bisect_right, bisect_left

class Multi_set:
    def __init__(self):
        self.total = 0
        self.ms = dict()
        self.lr = []

    def insert(self, x):
        self.total += 1
        if x in self.ms:
            self.ms[x] += 1
        else:
            self.ms[x] = 1
            insort(self.lr, x)
        print(self.total)

    def find(self, x):
        print(self.ms.get(x, 0))

    def delete(self, x):
        if x in self.ms:
            self.total -= self.ms[x]
            self.ms[x] = 0

    def dump(self, l, r):
        lb = bisect_left(self.lr, l)
        ub = bisect_right(self.lr, r)
        for i in range(lb, ub):
            k = self.lr[i]
            v = self.ms[k]
            print(f'{k}\\n' * v, end='')


ms = Multi_set()
for _ in range(int(input())):
    op, x, y = (input() + ' 1').split()[:3]
    if op == '0':
        ms.insert(int(x))
    elif op == '1':
        ms.find(int(x))
    elif op == '2':
        ms.delete(int(x))
    else:
        ms.dump(int(x), int(y))
'''

tc_22_in = '''
import bisect
MAX_N=2*10**5
isdmore5=[1 for i in range(MAX_N+1)]
isprime=[1 for i in range(MAX_N+1)]
i=2
isprime[0]=0;isprime[1]=0
isdmore5[0]=0;isdmore5[1]=0
if isprime[i]!=0:
    for j in range(2*i,MAX_N+1,i):
        isprime[j]=0

    i+=1
prime=[]
for i in range(MAX_N+1):
    if isprime[i]==1:
        prime.append(i)
        isdmore5[i]=0
#print(prime)
for p in prime:
    k=bisect.bisect_left(prime,MAX_N//p)
    #print(p,k,prime[k-1]*p,prime[k]*p,prime[k+1]*p)
    for i in range(k):
        #print(p,prime[i],p*prime[i])
        isdmore5[p*prime[i]]=0

i=2
if isprime[i]==1:
    isdmore5[i*i*i]=0

    i+=1
ans=[0 for i in range(MAX_N+1)]
for i in range(3,MAX_N+1):
    if isdmore5[i]==1:
        ans[i]=ans[i-1]+1
    else:
        ans[i]=ans[i-1]

Q=int(input())
for i in range(Q):
    print(ans[int(input())])'''
    
tc_22_out = '''
import bisect
MAX_N=2*10**5
isdmore5=[1 for i in range(MAX_N+1)]
isprime=[1 for i in range(MAX_N+1)]
i=2
isprime[0]=0;isprime[1]=0
isdmore5[0]=0;isdmore5[1]=0
if isprime[i]!=0:
    for j in range(2*i,MAX_N+1,i):
        isprime[j]=0

    i+=1
prime=[]
for i in range(MAX_N+1):
    if isprime[i]==1:
        prime.append(i)
        isdmore5[i]=0
#print(prime)
for p in prime:
    k=bisect.bisect_left(prime,MAX_N//p)
    #print(p,k,prime[k-1]*p,prime[k]*p,prime[k+1]*p)
    for i in range(k):
        #print(p,prime[i],p*prime[i])
        isdmore5[p*prime[i]]=0

i=2
if isprime[i]==1:
    isdmore5[i*i*i]=0

    i+=1
ans=[0 for i in range(MAX_N+1)]
for i in range(3,MAX_N+1):
    if isdmore5[i]==1:
        ans[i]=ans[i-1]+1
    else:
        ans[i]=ans[i-1]

Q=int(input())
for i in range(Q):
    print(ans[int(input())])'''
    
    
tc_23_in = '''
e, y = map(int, input().split())
if e == 0:
    if y < 1912:print("M" + str(y - 1867))
    elif y < 1926:print("T" + str(y - 1911))
    elif y < 1989:print("S" + str(y - 1925))
    else:print("H" + str(y - 1988))
else:print(y + (1867,1911,1925,1988)[e - 1])
'''

tc_23_out = '''
e, y = map(int, input().split())
if e == 0:
    if y < 1912:print("M" + str(y - 1867))
    elif y < 1926:print("T" + str(y - 1911))
    elif y < 1989:print("S" + str(y - 1925))
    else:print("H" + str(y - 1988))
else:print(y + (1867,1911,1925,1988)[e - 1])
'''

tc_24_in = '''
#標準入力
l = list(map(int,input().split()))

#リストを小さい順に並べる
l.sort()

#4辺を1人組とし、1組の辺が全て同じであれば"yes"でなければ"no"を出力する
if l[0] == l[1] == l[2] == l[3] and l[4] == l[5] == l[6] == l[7] and l[8] == l[9] == l[10] == l[11]:print("yes")
else:print("no")
'''

tc_24_out = '''
#標準入力
l = list(map(int,input().split()))

#リストを小さい順に並べる
l.sort()

#4辺を1人組とし、1組の辺が全て同じであれば"yes"でなければ"no"を出力する
if l[0] == l[1] == l[2] == l[3] and l[4] == l[5] == l[6] == l[7] and l[8] == l[9] == l[10] == l[11]:print("yes")
else:print("no")
'''

tc_25_in = '''
import bisect
MAX_N=2*10**5
isdmore5=[1 for i in range(MAX_N+1)]
isprime=[1 for i in range(MAX_N+1)]
i=2
isprime[0]=0;isprime[1]=0
isdmore5[0]=0;isdmore5[1]=0
while(i*i<=MAX_N):
    if isprime[i]!=0:
        for j in range(2*i,MAX_N+1,i):
            isprime[j]=0
    i+=1
prime=[]
for i in range(MAX_N+1):
    if isprime[i]==1:
        prime.append(i)
        isdmore5[i]=0
#print(prime)
for p in prime:
    k=bisect.bisect_left(prime,MAX_N//p)
    #print(p,k,prime[k-1]*p,prime[k]*p,prime[k+1]*p)
    for i in range(k):
        #print(p,prime[i],p*prime[i])
        isdmore5[p*prime[i]]=0
i=2
while(i*i*i<=MAX_N):
    if isprime[i]==1:
        isdmore5[i*i*i]=0
    i+=1
ans=[0 for i in range(MAX_N+1)]
for i in range(3,MAX_N+1):
    if isdmore5[i]==1:
        ans[i]=ans[i-1]+1
    else:
        ans[i]=ans[i-1]
Q=int(input())
for i in range(Q):
    print(ans[int(input())])
'''

tc_25_out = '''
import bisect
MAX_N=2*10**5
isdmore5=[1 for i in range(MAX_N+1)]
isprime=[1 for i in range(MAX_N+1)]
i=2
isprime[0]=0;isprime[1]=0
isdmore5[0]=0;isdmore5[1]=0
while(i*i<=MAX_N):
    if isprime[i]!=0:
        for j in range(2*i,MAX_N+1,i):
            isprime[j]=0
    i+=1
prime=[]
for i in range(MAX_N+1):
    if isprime[i]==1:
        prime.append(i)
        isdmore5[i]=0
#print(prime)
for p in prime:
    k=bisect.bisect_left(prime,MAX_N//p)
    #print(p,k,prime[k-1]*p,prime[k]*p,prime[k+1]*p)
    for i in range(k):
        #print(p,prime[i],p*prime[i])
        isdmore5[p*prime[i]]=0
i=2
while(i*i*i<=MAX_N):
    if isprime[i]==1:
        isdmore5[i*i*i]=0
    i+=1
ans=[0 for i in range(MAX_N+1)]
for i in range(3,MAX_N+1):
    if isdmore5[i]==1:
        ans[i]=ans[i-1]+1
    else:
        ans[i]=ans[i-1]
Q=int(input())
for i in range(Q):
    print(ans[int(input())])
'''
    




import difflib


tc_pairs = [
    (tc_1_in, tc_1_out),
    (tc_2_in, tc_2_out),
    (tc_3_in, tc_3_out),
    # (tc_4_in, tc_4_out),
    (tc_5_in, tc_5_out),
    (tc_6_in, tc_6_out),
    (tc_7_in, tc_7_out),
    (tc_8_in, tc_8_out),
    (tc_9_in, tc_9_out),
    (tc_10_in, tc_10_out),
    (tc_11_in, tc_11_out),
    (tc_12_in, tc_12_out), 
    (tc_13_in, tc_13_out),
    (tc_14_in, tc_14_out),
    (tc_15_in, tc_15_out),
    (tc_16_in, tc_16_out),  
    (tc_17_in, tc_17_out), 
    (tc_18_in, tc_18_out), 
    (tc_19_in, tc_19_out), 
    (tc_20_in, tc_20_out),
    (tc_21_in, tc_21_out), 
    (tc_22_in, tc_22_out),
    (tc_23_in, tc_23_out), 
    (tc_24_in, tc_24_out),
    (tc_25_in, tc_25_out)
]
import re 
if __name__ == '__main__':
    
    for i, (tc_in, tc_out) in enumerate(tc_pairs):
        # skip test case 10
        # if i == 9 or i ==10: 
        #     continue
        print(f"Running Test Case {i+1}")
        result = extract_python_code(tc_in)
        
        # if i not in (5, 8): 
        if True: 
            canonicalized_result = re.sub("\n+", "\n", result.strip())
            canonicalized_tc_out = re.sub("\n+", "\n", tc_out.strip())
            if not canonicalized_result.strip() == canonicalized_tc_out.strip():
                # diff = difflib.unified_diff(tc_out.strip().splitlines(), result.strip().splitlines(), lineterm='')
                diff = difflib.unified_diff(canonicalized_tc_out.strip().splitlines(), canonicalized_result.strip().splitlines(), lineterm='')
                print('\n'.join(diff))
            assert canonicalized_result.strip() == canonicalized_tc_out.strip(), f"Test Case {i+1} failed. Expected:\n{tc_out.strip()}\nGot:\n{result.strip()}"
            # assert result.strip() == tc_out.strip(), f"Test Case {i+1} failed. Expected:\n{tc_out.strip()}\nGot:\n{result.strip()}"
        else:
            print("Test Case 6 {i+1} is a special case, skipping assertion")
            print(f"Expected:\n{tc_out.strip()}")
            print(f"Got:\n{result.strip()}")
        
        print(f"Test Case {i+1} passed.")
        
    