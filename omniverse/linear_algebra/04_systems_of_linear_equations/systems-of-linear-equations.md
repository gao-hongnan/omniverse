Link back to
[Systems of Linear Equations](../../01_preliminaries/02-systems-of-linear-equations.md)
and use {eq}`02-systems-of-linear-equations-definition-algebraic-form-eq-1` as
the general form of a system of linear equations in
{prf:ref}`02-systems-of-linear-equations-definition-algebraic-form`

Check here for some inspiration:
https://github.com/weijie-chen/Linear-Algebra-With-Python/blob/master/Chapter%201%20-%20Linear%20Equation%20System.ipynb

### A Refresher on How to Plot a Plane in 3D Space

Probably useful when students know what is column space?

Let's say you want to plot the column space of a matrix

$$\mathbf{A} = \begin{bmatrix} 3 & -1 \\ 2 & 4 \\ -1 & 1 \end{bmatrix}$$

where the column space of $\mathbf{A}$ is just the span of the columns:

$$
\text{col}(\mathbf{A})=\text{span}\left\{\left[ \matrix{3\cr 2\cr -1}\right],\ \left[\matrix{-1\cr 4\cr 1}\right]\right\}
$$

then it follows that since the two column vectors are linearly independent, then
the span or rather the column space of $\mathbf{A}$ is the **set** of points
that make up a plane:

$$\text{col}(\mathbf{A}) = \text{plane} = \{s\left[\matrix{3\cr 2\cr -1}\right] + t\left[\matrix{-1\cr 4\cr 1}\right] | s, t \in \mathbb{R} \}$$

Then, to plot it, we can express the X, Y and Z components as follows:

$$X = 3s - t \quad Y = 2s + 4t \quad Z = -s + t$$

```{code-cell} ipython3
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(projection='3d')

s = np.linspace(-2, 2, 20)
t = np.linspace(-2, 2, 20)
S, T = np.meshgrid(s, t)

X = 3*S - T
Y = 2*S + 4*T
Z = -S + T
ax.plot_wireframe(X, Y, Z, linewidth = .5, color = 'r')
```

```{code-cell} ipython3
x1 = np.linspace(25, 35, 20)
x2 = np.linspace(10, 20, 20)
X1, X2 = np.meshgrid(x1, x2)

fig = plt.figure(figsize = (9, 9))
ax = fig.add_subplot(111, projection = '3d')

X3 = 2*X2 - X1
ax.plot_surface(X1, X2, X3, cmap ='viridis', alpha = 1)

X3 = .25*X2 - 1
ax.plot_surface(X1, X2, X3, cmap ='summer', alpha = 1)

X3 = -5/9*X2 + 4/9*X1 - 1
ax.plot_surface(X1, X2, X3, cmap ='spring', alpha = 1)

ax.scatter(29, 16, 3, s = 200, color = 'black')
plt.show()
```

ALso mention:

### Matrix Definition (System of Linear Equations)

The vector equation is equivalent to a matrix equation of the form
$\mathbf{A}\mathbf{x} = \mathbf{b}$, where
$\mathbf{A} \in \mathbb{F}^{m \times n}$, $\mathbf{x}$ a column vector in
$\mathbb{F}^n$ and $\mathbf{b}$ a column vector in $\mathbb{F}^m$.

$$
\mathbf{A} =
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix},\quad
\mathbf{x}=
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix},\quad
\mathbf{b}=
\begin{bmatrix}
b_1 \\
b_2 \\
\vdots \\
b_m
\end{bmatrix}
$$

### Vector Definition (System of Linear Equations)

Recall in the chapter on Matrix Multiplication, we note that
$\mathbf{A}\mathbf{x} = \mathbf{b}$ is a right multiplication of a matrix
$\mathbf{A}$ on the vector $\mathbf{b}$, and thus $\mathbf{b}$ can be
represented as the **linear combination of columns of $\mathbf{A}$ with $x_i$ as
coefficients**.

$$
\mathbf{b} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + ... + x_n \mathbf{a}_n \implies
\begin{bmatrix}
b_1 \\
b_2 \\
\vdots \\
b_m
\end{bmatrix} = x_1 \begin{bmatrix} a_{11} \\ a_{21} \\ \vdots \\ a_{m1} \end{bmatrix} + x_2 \begin{bmatrix} a_{12} \\ a_{22} \\ \vdots \\ a_{m2} \end{bmatrix} + ... + x_n \begin{bmatrix} a_{1n} \\ a_{2n} \\ \vdots \\ a_{mn} \end{bmatrix}
$$

### Definition (Homogeneous System of Equations)

A system of equations is called **homogeneous** if each equation in the system
is equal to $0$ . A homogeneous system has the form:

$$
\begin{align}
a_{11} x_1 + a_{12} x_2  + \cdots + a_{1n} x_n  &= 0 \\
a_{21} x_1 + a_{22} x_2  + \cdots + a_{2n} x_n  &= 0 \\
& \ \ \vdots\\
a_{m1} x_1 + a_{m2} x_2  + \cdots + a_{mn} x_n  &= 0,
\end{align}
$$

where $x_1, x_2,\ldots,x_n$ are the unknowns, $a_{11},a_{12},\ldots,a_{mn}$ are
the coefficients of the system.

> Note that this definition can be similarly translated in terms of Matrix and
> Vector definitions.

### Definition (Inconsistent and Consistent Systems)

-   **Consistent**: A system of linear equations are called **consistent** if
    there exists at least one solution.
-   **Inconsistent**: A system of linear equations are called **inconsistent**
    if there exists no solution.

## Elementary Row Operations

Elementarty row operations provide us a way to find out if a system of linear
equations is **consistent or not**.

### Definition (Elementary Row Operations)

In order to enable us to convert a system of linear equations to an
**equivalent** system, we define the following **elementary row operations**:

-   **Row Permutation:** Interchange any two rows of a matrix: $\r_i \iff \r_j$
-   **Row Multiply:** Replace any row of a matrix with a non-zero scalar
    multiple of itself: $\r_i \to \lambda\r_i$
-   **Row Addition:** Replace any row of a matrix with the sum of itself and a
    non-zero scalar multiple of any other row: $\r_i \to \r_i + \lambda \r_j$.

**$\r_i$ refers to row $i$ of the matrix.**

### Definition (Elementary Column Operations)

By replacing the word _row_ to _column_, we recover the definition of
**elementary column operations**.

### Theorem (Elementary Row Operations Preserve Solution Set of Linear Systems)

This theorem will be proven again later in the context of matrices. Here, I
highly recommend reading the proof (without the context of matrices) from
[A First Course in Linear Algebra by Ken Kuttler](<https://math.libretexts.org/Bookshelves/Linear_Algebra/A_First_Course_in_Linear_Algebra_(Kuttler)/01%3A_Systems_of_Equations/1.02%3A_Elementary_Operations>)
where he showed that these 3 operations will not change the solution set of the
original system of linear equations.

## Gauss Elimination

### Definition (Augmented Matrix of a System of Linear Equations)

We usually combine $\mathbf{A}\mathbf{x} = \mathbf{b}$ into one system (matrix)
for ease of computing elementary row operations, after all, row operations are
always applied to the **whole system**.

Given the general form of the linear equations, the **augmented matrix** of the
system of equations is:

$$
[\mathbf{A} ~|~ \mathbf{b}] = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} & b_1 \\
a_{21} & a_{22} & \cdots & a_{2n} & b_2 \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn} & b_m
\end{bmatrix}
$$

### Theorem (Solving Augmented Matrix Solves the System of Linear Equations)

We established that row operations on a system of linear equations preserve the
orginal solution set, therefore we can apply row operations on the augmented
matrix to solve the solution.

### Definition (Row Echolon Form)

Given a matrix $\mathbf{A} \in \mathbb{F}^{m \times n}$ and
$\mathbf{b} \in \mathbb{F}^{m}$, then we say the augmented matrix
$[\mathbf{A} ~|~ \mathbf{b}]$ is in its **row echolon form** if:

-   Any rows that are all **zeros** must be at the bottom of the matrix, that is
    to say, all **zero row vectors** are grouped at the bottom.
-   The **leading coefficient (also called the pivot)** of a non-zero row is
    always strictly to the right of the leading coefficient of the row above it.
-   All entries in a column below a pivot are zeros.
-   Some textbooks require the leading coefficient to be 1.

### Definition (Reduced Row Echolon Form)

Given a matrix $\mathbf{A} \in \mathbb{F}^{m \times n}$ and
$\mathbf{b} \in \mathbb{F}^{m}$, then we say the augmented matrix
$[\mathbf{A} ~|~ \mathbf{b}]$ is in its **reduced row echolon form** if:

-   Any rows that are all **zeros** must be at the bottom of the matrix, that is
    to say, all **zero row vectors** are grouped at the bottom.
-   The **leading coefficient (also called the pivot)** of a non-zero row is
    always strictly to the right of the leading coefficient of the row above it.
-   All entries in a column below a pivot are zeros.
-   The leading coefficient to be 1.
-   All entries in a column above and below a leading entry are zero.

### Definition (Pivot Position and Pivot Column)

-   **Pivot Position:** A **pivot position** in a matrix is the location of a
    leading entry in the row-echelon form of a matrix.

-   **Pivot Column:** A **pivot column** is a column that contains a pivot
    position.

### Algorithm (Gaussian and Gaussian-Jordan Elimination)

> Entirely taken from
> [A First Course in Linear Algebra by Ken Kuttler](<https://math.libretexts.org/Bookshelves/Linear_Algebra/A_First_Course_in_Linear_Algebra_(Kuttler)>).

This algorithm provides a method for using row operations to take a matrix to
its reduced row-echelon form. We begin with the matrix in its original form.

1. Starting from the left, find the first nonzero column. This is the first
   pivot column, and the position at the top of this column is the first pivot
   position. Switch rows if necessary to place a nonzero number in the first
   pivot position.
2. Use row operations to make the entries below the first pivot position (in the
   first pivot column) equal to zero.
3. Ignoring the row containing the first pivot position, repeat steps 1 and 2
   with the remaining rows. Repeat the process until there are no more rows to
   modify.
4. Divide each nonzero row by the value of the leading entry, so that the
   leading entry becomes 1 . The matrix will then be in row-echelon form.

> The following step will carry the matrix from row-echelon form to reduced
> row-echelon form.

5. Moving from right to left, use row operations to create zeros in the entries
   of the pivot columns which are above the pivot positions. The result will be
   a matrix in reduced row-echelon form.

### Definition (Types of Solutions)

> Modified from
> [A First Course in Linear Algebra by Ken Kuttler](<https://math.libretexts.org/Bookshelves/Linear_Algebra/A_First_Course_in_Linear_Algebra_(Kuttler)>).

#### Definition (No Solution)

In the case where the system of equations has no solution, the row-echelon form
of the augmented matrix will have a row of the form:

$$
\left[\begin{array}{@{}ccc|c@{}}
0 & 0 & \cdots & b_i \\
\end{array}\right]
$$

That is to say, there exists a row with entirely zeros in $\mathbf{A}$ but the
corresponding output $\b_i \neq 0$.

#### Definition (One Unique Solution)

We use a small example as follows:

$$
\left[\begin{array}{@{}ccc|c@{}}
1 & 0 & 0 & b_1 \\
0 & 1 & 0 & b_2 \\
0 & 0 & 1 & b_3 \\
\end{array}\right]
$$

This system has unique solution as every column of the coefficient matrix is a
pivot column.

#### Definition (Infinitely Many Solutions)

We use a small example as follows.

In the case where the system of equations has infinitely many solutions, the
solution contains parameters. There will be columns of the coefficient matrix
which are not pivot columns. The following are examples of augmented matrices in
reduced row-echelon form for systems of equations with infinitely many
solutions.

$$
\left[\begin{array}{@{}ccc|c@{}}
1 & 0 & 0 & b_1 \\
0 & 1 & 0 & b_2 \\
0 & 0 & 0 & 0 \\
\end{array}\right]
$$

## Uniqueness of Reduced Row-Echolon Form

### Definition (Basis Variable)

Assume a augmented matrix system $[\mathbf{A} ~|~ \mathbf{b}]$ in **rref**, then
the variables (unknowns) $x_i$ is a **basic variable** if
$[\mathbf{A} ~|~ \mathbf{b}]$ has a leading 1 in column number $i$, in this
case, column $i$ is also a **pivot column**.

### Definition (Free Variable)

If the variable $x_i$ is not **basis**, then it is **free**.

### Definition (Free Column)

A **free column** is a column that does not contains a pivot position.

### Example (Basic and Free Variable)

This is best understood from an
[example taken from A First Course in Linear Algebra by Ken Kuttler](<https://math.libretexts.org/Bookshelves/Linear_Algebra/A_First_Course_in_Linear_Algebra_(Kuttler)>).

Consider the system:

$$
\begin{align}
x + 2y - z + w = 3 \\
x + y - z + w = 1 \\
x + 3y - z + w = 5
\end{align}
$$

we know that the augmented matrix is:

$$
\left[\begin{array}{@{}cccc|c@{}}
1 & 2 & -1 & 1 & 3 \\
0 & 1 & 0  & 0 & 2 \\
0 & 0 & 0 & 0 & 0\\
\end{array}\right]
$$

**Solution**

-   We always look out for the row with 1 variable to one solution (if it
    exists). In this case, it is $y = 2$. The perks of **rref** allows us to do
    this easily.
-   In the first row, it has
    $x + 2y - z + w = 3 \implies x + 4 - z + w = 3 \implies x = -1 + z - w$.
    -   Since the solution of $x$ depends on $z$ and $w$, we call $z$ and $w$
        the free variable and parameters as $z$ and $w$ can actually take on any
        value.
    -   Set $z = s$
    -   Set $w = t$

So the solution set can be described as:

$$
\begin{bmatrix}
x \\ y \\ z \\ w
\end{bmatrix}
=
\begin{bmatrix}
-1 + s - t \\ 2 \\ s \\ t
\end{bmatrix}
$$

and has **infinitely number of solutions**.

Here, the **free variables** are the parameters $z = s$ and $w = t$, and **basic
variables**.

### Sorting out the confusion (Basic and Free Variables)

From the example above, we can clearly see that free variables allow us to
assign any values to them. The above example seems obvious, but it isn't that
much if we have:

$$
\left[\begin{array}{@{}cccc|c@{}}
1 & 2 & 0 & -2 & 0 \\
0 & 0 & 1 & 2 & 0 \\
0 & 0 & 0 & 0 & 0\\
\end{array}\right]
$$

which translates to:

$$
\begin{aligned}
x_1 + 2x_2 + 0x_3 - 2x_4 = 0 \\
0 x_1 + 0x_2 + x_3 + 2x_4 = 0
\end{aligned}
$$

By definition $x_2$ and $x_4$ are free variables, and if you ever wonder why
$x_4$ is free (even though it is by definition), then you did not understand the
basics.

Consider simply:

$$
\begin{aligned}
x + y = 0 \\
2x + 2y = 0
\end{aligned}
$$

then it is obvious that this system reduces to only solving $x + y = 0$, in
which if you do **RREF**, the free variable is $y$. If you plot out the solution
set, this is just a straight line $x + y = 0$ that passes through the origin.
Then if you write it as $x = -y$, then this means $x$ depends on $y$, in which
$y$ can be any point on the line. Similarly, if we ignore the definition of free
variable, we can also write $y = -x$ and recover our favourite high school
equation of a line where $y$ depends on $x$ and $x$ being independent is allowed
to take on any values. But matrix theory now gives us a systematic way to
approach things, we just need to know that if our unknowns is more than the
equations, we are usually bound to have free variables.

### Word of Caution (Basic and Free Variables)

Note that since normal Gaussian Elimination **REF** is not unique, there can be
different free and basic variables for different **REF**. But you will see that
**RREF** guarantees uniqueness.

### Proposition (Basic and Free Variables)

If $x_i$ is a basic variable of a homogeneous system of linear equations, then
any solution of the system with $x_j=0$ for all those free variables $x_j$ with
$j>i$ must also have $x_i=0$.

> This is best understood by the previous example, note that we can denote:

$$
x = x_1, y = x_2, z = x_3, w = x_4
$$

and see that the free variables below $x_1$ cannot have $x_1 \neq 0$ inside.

### Lemma (Solutions and the Reduced Row-Echelon Form of a Matrix)

Let $\mathbf{A}$ and $\mathbf{B}$ be two distinct augmented matrices for two
homogeneous systems of $m$ equations in $n$ variables, such that $A$ and $B$ are
each in reduced row-echelon. Then, the two systems do not have exactly the same
solutions.

### Definition (Row Equivalence)

Two matrices $\mathbf{A}$ and $\mathbf{B}$ are **row equivalent** if one matrix
can be obtained from the other matrix by a **finite sequence of elementary row
operations**.

> Note that if $\mathbf{A}$ can be obtained by applying a sequence of elementary
> row operations on $\mathbf{B}$, then it follows that we just need to apply the
> sequence in reverse for $\mathbf{B}$ to get to $\mathbf{A}$.

### Theorem (Every Matrix is row equivalent to its RREF)

Every matrix $\mathbf{A} \in \mathbb{F}^{m \times n}$ is row equivalent to its
**RREF**.

### Theorem (Row Equivalent Augmented Matrices have the same solution set)

Given $[\mathbf{A} ~|~ \mathbf{b}]$ and $[\C ~|~ \d]$, if both are **row
equivalent** to each other, then the two linear systems have the same solution
sets.

### Theorem (RREF is Unique)

Every matrix $\mathbf{A}$ has a **RREF** and it is unique. To prove it one
should use **Lemma (Solutions and the Reduced Row-Echelon Form of a Matrix)**
and **Theorem (Row Equivalent Augmented Matrices have the same solution set)**.
See
[A First Course in Linear Algebra by Ken Kuttler](<https://math.libretexts.org/Bookshelves/Linear_Algebra/A_First_Course_in_Linear_Algebra_(Kuttler)>).

## Rank and Homogeneous Systems

The section talks about matrix rank in homogeneous systems. I felt it is better
mentioned again in matrix theory. So do visit there.

## Elementary Matrices

### Permutation Matrix

Row Exchange:

$$
\begin{align}
x_1- 2x_2+x_3&=0\\
2x_2-8x_3&=8\\
-4x_1+5x_2+9x_3&=-9
\end{align}
$$

vs

$$
\begin{align}
2x_2-8x_3&=8\\
x_1- 2x_2+x_3&=0\\
-4x_1+5x_2+9x_3&=-9
\end{align}
$$

has no difference, we just swapped row 1 and 2. We can do the same in matrix for
conveince.

Also, given

$$
\mathbf{P} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \\ \end{bmatrix}
,\quad
\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \\ \end{bmatrix}
$$

then

$$
\mathbf{P}\mathbf{A} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \\ \end{bmatrix} \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \\ \end{bmatrix} = \begin{bmatrix} 4 & 5 & 6 \\ 1 & 2 & 3 \\ 7 & 8 & 9 \\ \end{bmatrix}
$$

and notice that row 1 and 2 are swapped by the left multiplication of the
permutation matrix $\mathbf{P}$. Why did it worked?

Recall now

$$\mathbf{P}\mathbf{A} = \begin{bmatrix}\ \p_1 \\ \p_2 \\  \p_3 \end{bmatrix}\mathbf{A} = \begin{bmatrix}\p_1\mathbf{A} \\ \p_2\mathbf{A} \\ \p_3\mathbf{A} \end{bmatrix}$$

We just look at the first row of $\mathbf{P}\mathbf{A}$ given by
$\p_1\mathbf{A}$ which maps to the first row of $\mathbf{P}\mathbf{A}$.

$$\p_1\mathbf{A} = 0 \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} + 1 \begin{bmatrix} 4 & 5 & 6 \end{bmatrix} + 0 \begin{bmatrix} 7 & 8 & 9 \end{bmatrix} = \begin{bmatrix} 4 & 5 & 6 \end{bmatrix}$$

Then the rest is the same logic:

$$\p_2\mathbf{A} = 1 \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} + 0 \begin{bmatrix} 4 & 5 & 6  \end{bmatrix} + 0 \begin{bmatrix} 7 & 8 & 9 \end{bmatrix} = \begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$$

$$\p_3\mathbf{A} = 0 \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} + 0 \begin{bmatrix} 4 & 5 & 6  \end{bmatrix} + 1 \begin{bmatrix} 7 & 8 & 9  \end{bmatrix} = \begin{bmatrix} 7 & 8 & 9 \end{bmatrix}$$

We now see why through **Matrix Multiplication (Left row wise)** that the
**Permutation Matrix** works the way it is!

---

### Introduction to Systems of Linear Equations in the Context of Linear Regression

#### Motivation: The Essence of Linear Regression

Linear regression is a foundational technique in machine learning, serving as a
stepping stone to understanding more complex models. At its core, linear
regression involves fitting a line (or a hyperplane in higher dimensions) to a
set of data points to predict an outcome. This process fundamentally relies on
systems of linear equations.

#### Algebraic Definition (System of Linear Equations)

Recall that a general system of $m$ linear equations with $n$ unknowns can be
written as:

$$
\begin{align}
a_{11} x_1 + a_{12} x_2  + \cdots + a_{1n} x_n  &= b_1 \\
a_{21} x_1 + a_{22} x_2  + \cdots + a_{2n} x_n  &= b_2 \\
& \ \ \vdots\\
a_{m1} x_1 + a_{m2} x_2  + \cdots + a_{mn} x_n  &= b_m,
\end{align}
$$

where $x_1, x_2,\ldots,x_n$ are the unknowns, $a_{11},a_{12},\ldots,a_{mn}$ are
the coefficients of the system, and $b_1,b_2,\ldots,b_m$ are the constant terms.

#### Linear Regression as a System of Linear Equations

In a linear regression setting, we have a dataset with $m$ observations and $n$
features. The goal is to find the weights (or coefficients) that best predict
the target variable. Mathematically, this can be represented as:

$$
\begin{align}
w_1 x_{11} + w_2 x_{12} + \cdots + w_n x_{1n} &= y_1 \\
w_1 x_{21} + w_2 x_{22} + \cdots + w_n x_{2n} &= y_2 \\
& \ \ \vdots \\
w_1 x_{m1} + w_2 x_{m2} + \cdots + w_n x_{mn} &= y_m,
\end{align}
$$

where $w_1, w_2, \ldots, w_n$ are the weights we aim to determine, $x_{ij}$
represents the $j^{th}$ feature of the $i^{th}$ observation, and $y_i$ is the
target variable for the $i^{th}$ observation.

#### The Connection

The resemblance between the general form of linear equations and the linear
regression equations is evident. In both cases, we are looking to solve for
unknown variables that satisfy a set of linear constraints. In machine learning,
especially in linear regression, these solutions help us make predictions or
understand relationships between variables. This direct application in a field
as dynamic and impactful as machine learning demonstrates the enduring relevance
and power of linear algebra, particularly systems of linear equations.

## References and Further Readings

-   [Wikipedia: System of linear equations](https://en.wikipedia.org/wiki/System_of_linear_equations)
-   Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). _Mathematics for
    Machine Learning_. Cambridge University Press. (Chapter 3.1, Norms).
-   https://github.com/weijie-chen/Linear-Algebra-With-Python/blob/master/Chapter%201%20-%20Linear%20Equation%20System.ipynb
-   https://github.com/fastai/numerical-linear-algebra
-   [A First Course in Linear Algebra by Ken Kuttler](<https://math.libretexts.org/Bookshelves/Linear_Algebra/A_First_Course_in_Linear_Algebra_(Kuttler)>)
-   https://math.stackexchange.com/questions/1634411/why-adding-or-subtracting-linear-equations-finds-their-intersection-point
-   Dr Choo Yan Min's treatment of lines and planes are good.
-   https://tutorial.math.lamar.edu/classes/calciii/eqnsofplanes.aspx
-   https://www.tuitionkenneth.com/h2-maths-parametric-scalar-product-cartesian
