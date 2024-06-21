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





big_tc = '''
Creating unit tests for this function involves crafting input strings that simulate various common scenarios of interleaved Python code and natural language text. Here are some sample inputs for testing:

1. **Basic Function Definition Interleaved with Natural Language:**
   ```python
   text1 = """
   Here is a simple function:
   def add(x, y):
       return x + y

   This function simply adds two numbers.
   """
   ```

2. **Multiple Functions with Different Indentation Levels:**
   ```python
   text2 = """
   def outer_function(a):
       # This function wraps another function
       def inner_function(b):
           return b * 2
       return inner_function(a)

   # End of nested functions
   """
   ```

3. **Import Statements Mixed with Comments:**
   ```python
   text3 = """
   # Here are some imports
   import os
   import sys
   from datetime import datetime

   # The imports are above
   """
   ```

4. **Complex Block with `if __name__ == "__main__":`**
   ```python
   text4 = """
   if __name__ == "__main__":
       import os
       import sys
       # This is the main block
       print("This is a test.")
       for i in range(5):
           print(i)
   # Main block ends above
   """
   ```

5. **Mixed Python Code and Detailed Commentary:**
   ```python
   text5 = """
   # Starting the function
   def process_data(data):
       # Loop through data
       for item in data:
           # Process each item
           print(item)  # Print item

   # The function processes data
   """
   ```

6. **Code with Improper Dedenting That Should Cause Errors:**
   ```python
   text6 = """
   def incorrect_indent():
   print("This should not work")
   """
   ```

7. **Properly and Improperly Indented Code Blocks:**
   ```python
   text7 = """
   def first_function():
       print("Properly indented")
   
       def nested_function():
           print("Also properly indented")

   def second_function():
   print("Improperly indented")
   """
   ```

8. **Using Decorators and Comments:**
   ```python
   text8 = """
   @decorator
   def decorated_function():
       # This function is decorated
       pass

   # Just a comment above a function
   def plain_function():
       pass
   """
   ```

9. **Blocks with Logical Continuation Using Backslashes:**
   ```python
   text9 = """
   import os, sys, \
       datetime, re

   def wrapped_line_function():
       x = 1 + 2 + \
           3 + 4
       return x
   """
   ```

10. **Random Natural Language with Embedded Python Code:**
    ```python
    text10 = """
    As discussed in the meeting, please review the following code:
    def review_code():
        # This function needs review
        for i in range(10):
            print("Number:", i)
    # Review should focus on loop efficiency.
    """
    ```

These test cases simulate different realistic scenarios you might encounter when processing text that mixes Python code with natural language comments, descriptions, or unrelated text. You can use these inputs to verify the robustness and accuracy of your code extraction function, ensuring it correctly identifies and extracts Python code blocks while properly handling indentation and dedents.
'''


import difflib


if __name__ == '__main__':
    expected_tcs = [(tc_1_in, tc_1_out), (tc_2_in, tc_2_out), (tc_3_in, tc_3_out), (tc_4_in, tc_4_out), (tc_5_in, tc_5_out), (tc_6_in, tc_6_out), (tc_7_in, tc_7_out), (tc_8_in, tc_8_out), (tc_9_in, tc_9_out), (tc_10_in, tc_10_out), (tc_11_in, tc_11_out)]
    for i, (tc_in, tc_out) in enumerate(expected_tcs):
        print(f"Running Test Case {i+1}")
        result = extract_python_code(tc_in)
        
        # if i not in (5, 8): 
        if True: 
            if not result.strip() == tc_out.strip():
                diff = difflib.unified_diff(tc_out.strip().splitlines(), result.strip().splitlines(), lineterm='')
                print('\n'.join(diff))
            assert result.strip() == tc_out.strip(), f"Test Case {i+1} failed. Expected:\n{tc_out.strip()}\nGot:\n{result.strip()}"
        else:
            print("Test Case 6 {i+1} is a special case, skipping assertion")
            print(f"Expected:\n{tc_out.strip()}")
            print(f"Got:\n{result.strip()}")
        
        print(f"Test Case {i+1} passed.")
        
    # print("Running Big Test Case")
    # result = extract_python_code(big_tc)
    # print("extracted code, from here there should be no language\n")
    # print(result)