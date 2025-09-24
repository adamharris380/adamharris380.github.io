# ECON 7510
# Adam Harris, adamharris@cornell.edu
# Fall 2025
# Lecture 7: Python Introduction for Numerical Methods

# Python Introduction for Numerical Methods
# ==========================================
# This file introduces basic Python concepts for numerical methods.
# It is designed for complete beginners and will be explained line by line.

# 1. Comments
# Comments start with a `#` and are ignored by Python.
# Use them to explain your code or temporarily disable parts of it.
""" Multiline
comments are
surrounded by triple quotes

"""

# 2. Variables and Basic Operations
# Python is dynamically typed, so you don't need to declare variable types.

# Example: Assigning values to variables
x = 10
y = 6

# Basic arithmetic operations.
# Mostly what you'd expect:
addition = x + y      # Addition
subtraction = x - y   # Subtraction
multiplication = x * y # Multiplication
division = x / y      # Division (returns a float)
print("Addition:", addition)
print("Subtraction:", subtraction)
print("Multiplication:", multiplication)
print("Division:", division)


# A few less intuitive operations:
integer_division = x // y # Floor division (integer result)
print("Integer Division:", integer_division)

modulus = x % y       # Remainder
print("Modulus:", modulus)

something = x^y       # Exponentiation?
print("10^6 returns:", something)
# No, this is bitwise XOR, not exponentiation
# Exponentiation is done with `**`
power = x ** y        # Exponentiation
print("Power:", power)

# 3. Lists
# A list is a collection of items, which can be of different types.

numbers = [1, 2, 3, 4, 5]
print("List of numbers:", numbers)

# Access elements by index
print("Element 1:", numbers[1])

# Python indexing is 0-based
# So to [0] refers to the first element
print("First element:", numbers[0])




# Add elements to a list
numbers.append(6)
print("After appending 6:", numbers)

# A list's elements can be of different types:
numbers.append(7.0)
numbers.append("ABC")
numbers

# Subsetting a list
print("First three elements:", numbers[0:3])
print("First three elements:", numbers[:3])
print("Last three elements:", numbers[-3:])

# How long is a list?
len(numbers)

# Range: An array containing the first n natural numbers
numbers2 = range(6)
numbers2[0]
numbers2[1]
numbers2[2]
numbers2[5]
numbers2[6]
# This doesn't work because Python uses 0-based indexing, so the last index is n-1

# Accessing the last or second-to-last element
numbers2[-1]
numbers2[-2]

# Or an array with natural numbers from n to m:
numbers3 = range(2,5)
numbers3[0]
numbers3[1]
numbers3[2]
numbers3[3]
# This doesn't work because the range goes up to, but does not include, the stop value (5 in this case)

# 4. Functions
# Functions group reusable code. Define a function with `def`.
def square(num):
    return num ** 2

# Call the function
print("Square of 4:", square(4))

# Lambda functions
# Lambda functions are anonymous, single-line functions.

cube = lambda x: x ** 3
print("Cube of 3:", cube(3))

# 5. Loops
# Loops allow you to repeat a block of code multiple times.

# For loop
for number in numbers:
    print("Number in list:", number)

# While loop
counter = 0
while counter < 3:
    print("Counter:", counter)
    counter += 1

# For loop with range
for i in range(10):
    print(i)

# 6. Branching Statements
# Branching allows your code to make decisions based on conditions.

# Example: If-elif-else statements
def sign(num):
    if num > 0:
        print("The number is positive.")
    elif num == 0:
        print("The number is zero.")
    else:
        print("The number is negative.")

sign(0.1)
sign(-11)
sign(0)


""" TWO EXAMPLE PROBLEMS """
# Example problem: Given a natural number n, the function returns the product of all odd natural numbers less than or equal to n.
def example_one(n):
    prod = 1
    for i in range(1,n+1):
        if i % 2 == 1:
            prod *= i
    return prod

example_one(3)
example_one(4)
example_one(5)
example_one(10)
1 * 3 * 5 * 7 * 9


# Example problem: Given a natural number n, the function returns the largest Fibonacci number less than or equal to n.
# (The Fibonacci sequence is defined as follows: F0 = 1, F1 = 1 and for all k > 1, Fk = Fk−2 + Fk−1.)
def example_two(n):
    a, b = 1, 1
    while b <= n:
        a, b = b, a + b
    return a

example_two(7)
example_two(8)
example_two(20)
example_two(21)
example_two(22)
example_two(1000000)





# 7. Data Types and Conversions
# Python has various data types, and you can convert between them.

# Basic types
integer_var = 42
float_var = 3.14
string_var = "Hello, Python!"
boolean_var = True

type(integer_var)
type(float_var)
type(string_var)
type(boolean_var)

# Type conversion
int_to_float = float(42)  # Convert int to float
print("Integer to float:", int_to_float)
float_to_int = int(3.14)      # Convert float to int
print("Float to integer:", float_to_int)
float_to_int2 = int(3.9999)      # Convert float to int
print("Float to integer:", float_to_int2)
num_to_string = str(42)   # Convert number to string
42
num_to_string
string_to_int = int("123")         # Convert string to int (if valid)
print("String to integer:", string_to_int)



# Edge cases:
int(3.1)
int(3.9)
int(4.0)

int("abc")

# 8. Importing Libraries
# Python has many libraries for numerical methods. Import them as needed.

# You might think that the following functions are built-in, but they are not:
sqrt(16.0)
log(3.0)



import math  # Math library for mathematical functions

# Example: Using math functions
print("Square root of 16:", math.sqrt(16))
print("Log of 3:", math.log(3.0))
print("Value of pi:", math.pi)
print("Value of e:", math.e)

# 9. Random Number Generation
# Random numbers are useful for simulations and other numerical methods.

import numpy as np
# Convention: import numpy as np
# As we'll see later on, numpy is also very useful for vector/matrix operations

# Can also do basically all of the same things as in the math library...
print("Square root of 16:", np.sqrt(16))
print("Log of 3:", np.log(3.0))
print("Value of pi:", np.pi)
print("Value of e:", np.e)

# ...plus a lot more
# Generate random numbers
random_number = np.random.rand()  # Random float in [0, 1)
print("Random number between 0 and 1:", random_number)

random_integers = np.random.randint(0, 10, size=5)  # 5 random integers between 0 and 10
print("Random integers:", random_integers)

# Simulating a coin flip
coin_flip = np.random.choice(["Heads", "Tails"])
print("Coin flip result:", coin_flip)

# 10. numpy arrays (like lists, but better for numerical methods)
my_first_array = np.array([1,2,3,6,0])
my_first_array
type(my_first_array)

my_first_array = np.append(my_first_array, [27])
my_first_array

my_first_array = np.append(my_first_array, ["abcd"])
my_first_array

# Create arrays with zeros, ones, or a range of numbers
zeros = np.zeros(5)
ones = np.ones(5)
range_array = np.arange(0, 10, 2)  # Start, stop, step

print("Zeros array:", zeros)
print("Ones array:", ones)
print("Range array:", range_array)

# 11. Plotting with Matplotlib
# Matplotlib is a library for creating visualizations.

import matplotlib.pyplot as plt

# Example: Plotting a simple function
def f(x):
    return x ** 2

x_values = np.linspace(-10, 10, 100)  # 101 points between -10 and 10
x_values
y_values = f(x_values)
y_values

plt.plot(x_values, y_values, label="y = x^2")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Plot of y = x^2")
plt.legend()
plt.grid()
plt.show()

# 12. Linear Algebra with Numpy
# Numpy provides tools for linear algebra, which are essential in numerical methods.

# Example: Matrix multiplication
a = np.array([1, 2, 3, 4])
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("Matrix A:\n", A)
print("Matrix B:\n", B)

# Two kinds of multiplication for matrices:
C = A * B
D = A @ B
print("Matrix A * B:\n", C)
print("Matrix A @ B:\n", D)
# What operations is each of these performing?  Which is "actual" matrix multiplication?

# Example: Solving a linear system Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b)
print("Solution to Ax = b:", x)

# Equivalent method:
x = np.linalg.inv(A) @ b
print("Solution to Ax = b:", x)

# Example: Eigenvalues and eigenvectors
values, vectors = np.linalg.eig(A)
print("Eigenvalues:", values)
print("Eigenvectors:", vectors)

# End of Python Introduction for Numerical Methods

""" FOUR MORE EXAMPLE PROBLEMS (FOR YOU TO TRY YOURSELF) """
# Example problem: Given a real number b \in (0, 900), the function returns the solution to x^3 − x^2 = b on x \in [1, 10].
# Note: x^3 - x^2 is increasing on [1, 10].
def g(x):
    return x ** 3 - x ** 2

# Plotting
x = np.linspace(1,10,100)
plt.clf()
plt.plot(x, g(x))
plt.xlabel("x")
plt.ylabel("g(x)")
plt.show()

g(1.0)
g(10.0)

# Left as an exercise for you to write this function:
def example_three(b):
    # Your code here
    left, right = 1.0, 10.0
    while right - left > 1e-7:
        mid = (left + right) / 2
        if g(mid) < b:
            left = mid
        else:
            right = mid
    return (left + right) / 2


# TEST: If the function works correctly, for any b, g(example_three(b))-b should be very close to zero
test_function = lambda b: g(example_three(b)) - b
test_function(100 * math.pi)
test_function(123.456)

# And here's a bit of code to help you visualize the solution:
# Applying function element-wise to an array:
b_vec = np.arange(0, 900, 2.0)
x_vec = [example_three(b) for b in b_vec]

# Plotting
plt.clf()
plt.plot(b_vec, x_vec)
plt.xlabel("b")
plt.ylabel("x")
plt.show()


# Example problem: Pascal's Triangle:
# Write a function that takes as input a natural number n >= 1 and returns a list of length n,
# where the ith element is the ith row of Pascal's triangle.
# Example input:  pascal(4)
# Example output: [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1]]
# https://en.wikipedia.org/wiki/Pascal%27s_triangle
def pascal(n):
    out = [[1]]
    for i in range(1,n):
        newrow = []
        for j in range(i):
            if (j == 0) | (j == i):
                newrow.append(1)
            else:
                newrow.append(out[-1][j-1] + out[-1][j])
        newrow.append(1)
        out.append(newrow)
    return out

# Test cases:
pascal(4)
[[1], [1, 1], [1, 2, 1], [1, 3, 3, 1]]

pascal(10)
[[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1], [1, 5, 10, 10, 5, 1], [1, 6, 15, 20, 15, 6, 1], [1, 7, 21, 35, 35, 21, 7, 1], [1, 8, 28, 56, 70, 56, 28, 8, 1], [1, 9, 36, 84, 126, 126, 84, 36, 9, 1]]

max(max(pascal(100)))
50445672272782096667406248628


# Example problem: Monte Carlo approximation of pi
# Write a function that takes as input a natural number n >= 1 and returns an approximation of pi using the following algorithm:
# 1. Generate n random points (x,y) in the unit square [0,1] x [0,1].
# 2. Count how many of these points fall within the quarter circle of radius 1 centered at the origin.
# 3. The approximation of pi is then given by 4 times the ratio of points inside the quarter circle to the total number of points n.
def approximate_pi(n):
    points = np.random.rand(n, 2)  # Generate n random points in [0,1] x [0,1]
    inside_circle = np.sum(points[:, 0]**2 + points[:, 1]**2 <= 1)  # Count points inside the quarter circle
    pi_approx = 4 * inside_circle / n  # Approximate pi
    return pi_approx

# Test cases:
approximate_pi(1000) - np.pi # Should be close to 0
approximate_pi(10000) - np.pi # Should be closer to 0
approximate_pi(1000000) - np.pi # Should be even closer to 0



# Example problem: Approximating e^x using Taylor series
# Write a function that takes as input a real number x and a natural number n >= 1,
# and returns the Taylor series approximation of e^x up to the nth term.
# Hint: The Taylor series expansion of e^x around 0 is given by:
# e^x = sum_{i=0}^{infinity} (x^i) / i!
# where i! is the factorial of i.
# math.factorial(i) computes i! in Python.
def approximate_exponential(x, n):
    approx = 0
    for i in range(n):
        approx += x ** i / math.factorial(i)
    return approx

# Test cases: All of the below should be very close to zero
approximate_exponential(1, 10) - np.e
approximate_exponential(1, 20) - np.e

approximate_exponential(2, 20) - np.e**2
approximate_exponential(2, 50) - np.e**2

approximate_exponential(-5, 50) - np.e**-5
