from typing import List, Tuple, Union

import numpy as np


def linear_combination_vectors(weights: List[float], *args: np.ndarray) -> np.ndarray:
    """Computes the linear combination of vectors.

    Args:
        weights (List[float]): The set of weights corresponding to each vector.

    Returns:
        linear_weighted_sum (np.ndarray): The linear combination of vectors.

    Examples:
        >>> v1 = np.asarray([1, 2, 3, 4, 5]).reshape(-1, 1)
        >>> v2 = np.asarray([2, 4, 6, 8, 10]).reshape(-1, 1)
        >>> v3 = np.asarray([3, 6, 9, 12, 15]).reshape(-1, 1)
        >>> weights = [10, 20, 30]
        >>> linear_combination_vectors([10, 20, 30], v1, v2, v3)
    """

    linear_weighted_sum = np.zeros(shape=args[0].shape)
    for weight, vec in zip(weights, args):
        linear_weighted_sum += weight * vec
    return linear_weighted_sum


def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes the dot product of two vectors.

    We assume both vectors are flattened, i.e. they are 1D arrays.

    Args:
        v1 (np.ndarray): The first vector.
        v2 (np.ndarray): The second vector.

    Returns:
        dot_product_v1_v2 (float): The dot product of two vectors.

    Examples:
        >>> v1 = np.asarray([1, 2, 3, 4, 5])
        >>> v2 = np.asarray([2, 4, 6, 8, 10])
        >>> dot_product(v1, v2)
    """

    v1, v2 = np.asarray(v1).flatten(), np.asarray(v2).flatten()

    dot_product_v1_v2 = 0
    for element_1, element_2 in zip(v1, v2):
        dot_product_v1_v2 += element_1 * element_2

    # same as np.dot but does not take into the orientation of vectors
    assert dot_product_v1_v2 == np.dot(v1.T, v2)

    return dot_product_v1_v2


def average_set(vec: Union[np.ndarray, set]) -> float:
    """Average a set of numbers using dot product.

    Given a set of numbers {v1, v2, ..., vn}, the average is defined as:
    avg = (v1 + v2 + ... + vn) / n

    To use the dot product, we can convert the set to a col/row vector (array) `vec` and
    perform the dot product with the vector of ones to get `sum(set)`. Lastly, we divide by the number of elements in the set.

    Args:
        vec (Union[np.ndarray, set]): A set of numbers.

    Returns:
        average (float): The average of the set.

    Examples:
        >>> v = np.asarray([1, 2, 3, 4, 5])
        >>> v_set = {1,2,3,4,5} # same as v but as a set.
        >>> average = average_set(v_set)
        >>> average = 3.0
    """

    if isinstance(vec, set):
        vec = np.asarray(list(vec)).flatten()

    ones = np.ones(shape=vec.shape)
    total_sum = dot_product(vec, ones)
    average = total_sum / vec.shape[0]

    assert average == np.mean(vec)
    return average


def get_matmul_shape(A: np.ndarray, B: np.ndarray) -> Tuple[int, int, int]:
    """Check if the shape of the matrices A and B are compatible for matrix multiplication.

    If A and B are of size (m, n) and (n, p), respectively, then the shape of the resulting matrix is (m, p).

    Args:
        A (np.ndarray): The first matrix.
        B (np.ndarray): The second matrix.

    Raises:
        ValueError: Raises a ValueError if the shape of the matrices A and B are not compatible for matrix multiplication.

    Returns:
        (Tuple[int, int, int]): (m, n, p) where (m, n) is the shape of A and (n, p) is the shape of B.
    """

    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"The number of columns of A must be equal to the number of rows of B, but got {A.shape[1]} and {B.shape[0]} respectively."
        )

    return (A.shape[0], A.shape[1], B.shape[1])


def np_matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Computes the matrix multiplication of two matrices.

    Args:
        A (np.ndarray): The first matrix.
        B (np.ndarray): The second matrix.

    Returns:
        matmul (np.ndarray): The matrix multiplication of two matrices.
    """

    num_rows_A, common_index, num_cols_B = get_matmul_shape(A, B)

    matmul = np.zeros(shape=(num_rows_A, num_cols_B))

    # 1st loop: loops through first matrix A
    for i in range(num_rows_A):
        summation = 0
        # 2nd loop: loops through second matrix B
        for j in range(num_cols_B):
            # 3rd loop: computes dot prod
            for k in range(common_index):
                summation += A[i, k] * B[k, j]
                matmul[i, j] = summation

    return matmul


def np_matmul_element_wise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Computes the matrix multiplication of two matrices using element wise method.

    Args:
        A (np.ndarray): The first matrix.
        B (np.ndarray): The second matrix.

    Returns:
        matmul (np.ndarray): The matrix multiplication of two matrices.
    """

    num_rows_A, _, num_cols_B = get_matmul_shape(A, B)

    matmul = np.zeros(shape=(num_rows_A, num_cols_B))

    # 1st loop: loops through first matrix A
    for row_i in range(num_rows_A):
        # 2nd loop: loops through second matrix B
        for col_j in range(num_cols_B):
            # computes dot product of row i with column j of B and
            # assign the result to the element of the matrix matmul at row i and column j.
            matmul[row_i, col_j] = dot_product(A[row_i, :], B[:, col_j])

    return matmul


def np_matmul_column_wise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Computes the matrix multiplication of two matrices using column wise method.

    Recall the section on Matrix Multiplication using Right Multiplication.

    Column i of C is represented by: Ab_i

    Args:
        A (np.ndarray): The first matrix.
        B (np.ndarray): The second matrix.

    Returns:
        matmul (np.ndarray): The matrix multiplication of two matrices.
    """
    num_rows_A, _, num_cols_B = get_matmul_shape(A, B)

    matmul = np.zeros(shape=(num_rows_A, num_cols_B))

    # we just need to populate the columns of C

    for col_i in range(matmul.shape[1]):
        # b_i
        col_i_B = B[:, col_i]
        # Ab_i
        linear_comb_A_on_col_i_B = linear_combination_vectors(col_i_B, *A.T)
        # C_i = Ab_i
        matmul[:, col_i] = linear_comb_A_on_col_i_B

    return matmul


import matplotlib.pyplot as plt
import numpy as np


def linearCombo(a, b, c):
    """This function is for visualizing linear combination of standard basis in 3D.
    Function syntax: linearCombo(a, b, c), where a, b, c are the scalar multiplier,
    also the elements of the vector.

    https://github.com/MacroAnalyst/Linear_Algebra_With_Python/blob/master/Chapter%209%20-%20Basis%20and%20Dimension.ipynb
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")

    ######################## Standard basis and Scalar Multiplid Vectors#########################
    vec = np.array(
        [
            [[0, 0, 0, 1, 0, 0]],  # e1
            [[0, 0, 0, 0, 1, 0]],  # e2
            [[0, 0, 0, 0, 0, 1]],  # e3
            [[0, 0, 0, a, 0, 0]],  # a* e1
            [[0, 0, 0, 0, b, 0]],  # b* e2
            [[0, 0, 0, 0, 0, c]],  # c* e3
            [[0, 0, 0, a, b, c]],
        ]
    )  # ae1 + be2 + ce3
    colors = ["b", "b", "b", "r", "r", "r", "g"]
    for i in range(vec.shape[0]):
        X, Y, Z, U, V, W = zip(*vec[i, :, :])
        ax.quiver(
            X,
            Y,
            Z,
            U,
            V,
            W,
            length=1,
            normalize=False,
            color=colors[i],
            arrow_length_ratio=0.08,
            pivot="tail",
            linestyles="solid",
            linewidths=3,
            alpha=0.6,
        )

    #################################Plot Rectangle Boxes##############################
    dlines = np.array(
        [
            [[a, 0, 0], [a, b, 0]],
            [[0, b, 0], [a, b, 0]],
            [[0, 0, c], [a, b, c]],
            [[0, 0, c], [a, 0, c]],
            [[a, 0, c], [a, b, c]],
            [[0, 0, c], [0, b, c]],
            [[0, b, c], [a, b, c]],
            [[a, 0, 0], [a, 0, c]],
            [[0, b, 0], [0, b, c]],
            [[a, b, 0], [a, b, c]],
        ]
    )
    colors = ["k", "k", "g", "k", "k", "k", "k", "k", "k"]
    for i in range(dlines.shape[0]):
        ax.plot(
            dlines[i, :, 0],
            dlines[i, :, 1],
            dlines[i, :, 2],
            lw=3,
            ls="--",
            color="black",
            alpha=0.5,
        )

    #################################Annotation########################################
    ax.text(x=a, y=b, z=c, s=" $(%0.d, %0.d, %.0d)$" % (a, b, c), size=18)
    ax.text(x=a, y=0, z=0, s=" $%0.d e_1 = (%0.d, 0, 0)$" % (a, a), size=15)
    ax.text(x=0, y=b, z=0, s=" $%0.d e_2 = (0, %0.d, 0)$" % (b, b), size=15)
    ax.text(x=0, y=0, z=c, s=" $%0.d e_3 = (0, 0, %0.d)$" % (c, c), size=15)

    #################################Axis Setting######################################
    ax.grid()
    ax.set_xlim([0, a + 1])
    ax.set_ylim([0, b + 1])
    ax.set_zlim([0, c + 1])

    ax.set_xlabel("x-axis", size=18)
    ax.set_ylabel("y-axis", size=18)
    ax.set_zlabel("z-axis", size=18)

    ax.set_title("Vector $(%0.d, %0.d, %.0d)$ Visualization" % (a, b, c), size=20)

    ax.view_init(elev=20.0, azim=15)


def linearComboNonStd(a, b, c, vec1, vec2, vec3):
    """This function is for visualizing linear combination of non-standard basis in 3D.
    Function syntax: linearCombo(a, b, c, vec1, vec2, vec3), where a, b, c are the scalar multiplier,
    ve1, vec2 and vec3 are the basis.

    https://github.com/MacroAnalyst/Linear_Algebra_With_Python/blob/master/Chapter%209%20-%20Basis%20and%20Dimension.ipynb
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    ########################Plot basis##############################
    vec1 = np.array([[0, 0, 0, vec1[0], vec1[1], vec1[2]]])
    X, Y, Z, U, V, W = zip(*vec1)
    ax.quiver(
        X,
        Y,
        Z,
        U,
        V,
        W,
        length=1,
        normalize=False,
        color="blue",
        arrow_length_ratio=0.08,
        pivot="tail",
        linestyles="solid",
        linewidths=3,
    )

    vec2 = np.array([[0, 0, 0, vec2[0], vec2[1], vec2[2]]])
    X, Y, Z, U, V, W = zip(*vec2)
    ax.quiver(
        X,
        Y,
        Z,
        U,
        V,
        W,
        length=1,
        normalize=False,
        color="blue",
        arrow_length_ratio=0.08,
        pivot="tail",
        linestyles="solid",
        linewidths=3,
    )

    vec3 = np.array([[0, 0, 0, vec3[0], vec3[1], vec3[2]]])
    X, Y, Z, U, V, W = zip(*vec3)
    ax.quiver(
        X,
        Y,
        Z,
        U,
        V,
        W,
        length=1,
        normalize=False,
        color="blue",
        arrow_length_ratio=0.08,
        pivot="tail",
        linestyles="solid",
        linewidths=3,
    )

    ###########################Plot Scalar Muliplied Vectors####################
    avec1 = a * vec1
    X, Y, Z, U, V, W = zip(*avec1)
    ax.quiver(
        X,
        Y,
        Z,
        U,
        V,
        W,
        length=1,
        normalize=False,
        color="red",
        alpha=0.6,
        arrow_length_ratio=a / 100,
        pivot="tail",
        linestyles="solid",
        linewidths=3,
    )

    bvec2 = b * vec2
    X, Y, Z, U, V, W = zip(*bvec2)
    ax.quiver(
        X,
        Y,
        Z,
        U,
        V,
        W,
        length=1,
        normalize=False,
        color="red",
        alpha=0.6,
        arrow_length_ratio=b / 100,
        pivot="tail",
        linestyles="solid",
        linewidths=3,
    )

    cvec3 = c * vec3
    X, Y, Z, U, V, W = zip(*cvec3)
    ax.quiver(
        X,
        Y,
        Z,
        U,
        V,
        W,
        length=1,
        normalize=False,
        color="red",
        alpha=0.6,
        arrow_length_ratio=c / 100,
        pivot="tail",
        linestyles="solid",
        linewidths=3,
    )

    combo = avec1 + bvec2 + cvec3
    X, Y, Z, U, V, W = zip(*combo)
    ax.quiver(
        X,
        Y,
        Z,
        U,
        V,
        W,
        length=1,
        normalize=False,
        color="green",
        alpha=0.7,
        arrow_length_ratio=np.linalg.norm(combo) / 300,
        pivot="tail",
        linestyles="solid",
        linewidths=3,
    )

    #################################Plot Rectangle Boxes##############################
    point1 = [avec1[0, 3], avec1[0, 4], avec1[0, 5]]
    point2 = [
        avec1[0, 3] + bvec2[0, 3],
        avec1[0, 4] + bvec2[0, 4],
        avec1[0, 5] + bvec2[0, 5],
    ]
    line1 = np.array([point1, point2])
    ax.plot(
        line1[:, 0],
        line1[:, 1],
        line1[:, 2],
        lw=3,
        ls="--",
        color="black",
        alpha=0.5,
    )

    point1 = [bvec2[0, 3], bvec2[0, 4], bvec2[0, 5]]
    point2 = [
        avec1[0, 3] + bvec2[0, 3],
        avec1[0, 4] + bvec2[0, 4],
        avec1[0, 5] + bvec2[0, 5],
    ]
    line1 = np.array([point1, point2])
    ax.plot(
        line1[:, 0],
        line1[:, 1],
        line1[:, 2],
        lw=3,
        ls="--",
        color="black",
        alpha=0.5,
    )

    point1 = [bvec2[0, 3], bvec2[0, 4], bvec2[0, 5]]
    point2 = [
        cvec3[0, 3] + bvec2[0, 3],
        cvec3[0, 4] + bvec2[0, 4],
        cvec3[0, 5] + bvec2[0, 5],
    ]
    line1 = np.array([point1, point2])
    ax.plot(
        line1[:, 0],
        line1[:, 1],
        line1[:, 2],
        lw=3,
        ls="--",
        color="black",
        alpha=0.5,
    )

    point1 = [cvec3[0, 3], cvec3[0, 4], cvec3[0, 5]]
    point2 = [
        cvec3[0, 3] + bvec2[0, 3],
        cvec3[0, 4] + bvec2[0, 4],
        cvec3[0, 5] + bvec2[0, 5],
    ]
    line1 = np.array([point1, point2])
    ax.plot(
        line1[:, 0],
        line1[:, 1],
        line1[:, 2],
        lw=3,
        ls="--",
        color="black",
        alpha=0.5,
    )

    point1 = [cvec3[0, 3], cvec3[0, 4], cvec3[0, 5]]
    point2 = [
        cvec3[0, 3] + avec1[0, 3],
        cvec3[0, 4] + avec1[0, 4],
        cvec3[0, 5] + avec1[0, 5],
    ]
    line1 = np.array([point1, point2])
    ax.plot(
        line1[:, 0],
        line1[:, 1],
        line1[:, 2],
        lw=3,
        ls="--",
        color="black",
        alpha=0.5,
    )

    point1 = [avec1[0, 3], avec1[0, 4], avec1[0, 5]]
    point2 = [
        cvec3[0, 3] + avec1[0, 3],
        cvec3[0, 4] + avec1[0, 4],
        cvec3[0, 5] + avec1[0, 5],
    ]
    line1 = np.array([point1, point2])
    ax.plot(
        line1[:, 0],
        line1[:, 1],
        line1[:, 2],
        lw=3,
        ls="--",
        color="black",
        alpha=0.5,
    )

    ##
    point1 = [
        avec1[0, 3] + bvec2[0, 3] + cvec3[0, 3],
        avec1[0, 4] + bvec2[0, 4] + cvec3[0, 4],
        avec1[0, 5] + bvec2[0, 5] + cvec3[0, 5],
    ]
    point2 = [
        cvec3[0, 3] + avec1[0, 3],
        cvec3[0, 4] + avec1[0, 4],
        cvec3[0, 5] + avec1[0, 5],
    ]
    line1 = np.array([point1, point2])
    ax.plot(
        line1[:, 0],
        line1[:, 1],
        line1[:, 2],
        lw=3,
        ls="--",
        color="black",
        alpha=0.5,
    )

    ##
    point1 = [
        avec1[0, 3] + bvec2[0, 3] + cvec3[0, 3],
        avec1[0, 4] + bvec2[0, 4] + cvec3[0, 4],
        avec1[0, 5] + bvec2[0, 5] + cvec3[0, 5],
    ]
    point2 = [
        cvec3[0, 3] + bvec2[0, 3],
        cvec3[0, 4] + bvec2[0, 4],
        cvec3[0, 5] + bvec2[0, 5],
    ]
    line1 = np.array([point1, point2])
    ax.plot(
        line1[:, 0],
        line1[:, 1],
        line1[:, 2],
        lw=3,
        ls="--",
        color="black",
        alpha=0.5,
    )

    ##
    point1 = [
        avec1[0, 3] + bvec2[0, 3] + cvec3[0, 3],
        avec1[0, 4] + bvec2[0, 4] + cvec3[0, 4],
        avec1[0, 5] + bvec2[0, 5] + cvec3[0, 5],
    ]
    point2 = [
        bvec2[0, 3] + avec1[0, 3],
        bvec2[0, 4] + avec1[0, 4],
        bvec2[0, 5] + avec1[0, 5],
    ]
    line1 = np.array([point1, point2])
    ax.plot(
        line1[:, 0],
        line1[:, 1],
        line1[:, 2],
        lw=3,
        ls="--",
        color="black",
        alpha=0.5,
    )
    #################################Annotation########################################
    ax.text(
        x=vec1[0, 3],
        y=vec1[0, 4],
        z=vec1[0, 5],
        s=" $v_1 =(%0.d, %0.d, %.0d)$" % (vec1[0, 3], vec1[0, 4], vec1[0, 4]),
        size=8,
    )
    ax.text(
        x=vec2[0, 3],
        y=vec2[0, 4],
        z=vec2[0, 5],
        s=" $v_2 =(%0.d, %0.d, %.0d)$" % (vec2[0, 3], vec2[0, 4], vec2[0, 4]),
        size=8,
    )
    ax.text(
        x=vec3[0, 3],
        y=vec3[0, 4],
        z=vec3[0, 5],
        s=" $v_3= (%0.d, %0.d, %.0d)$" % (vec3[0, 3], vec3[0, 4], vec3[0, 4]),
        size=8,
    )

    ax.text(
        x=avec1[0, 3],
        y=avec1[0, 4],
        z=avec1[0, 5],
        s=" $%.0d v_1 =(%0.d, %0.d, %.0d)$" % (a, avec1[0, 3], avec1[0, 4], avec1[0, 4]),
        size=8,
    )
    ax.text(
        x=bvec2[0, 3],
        y=bvec2[0, 4],
        z=bvec2[0, 5],
        s=" $%.0d v_2 =(%0.d, %0.d, %.0d)$" % (b, bvec2[0, 3], bvec2[0, 4], bvec2[0, 4]),
        size=8,
    )
    ax.text(
        x=cvec3[0, 3],
        y=cvec3[0, 4],
        z=cvec3[0, 5],
        s=" $%.0d v_3= (%0.d, %0.d, %.0d)$" % (c, cvec3[0, 3], cvec3[0, 4], cvec3[0, 4]),
        size=8,
    )
    #     ax.text(x = 0, y = b, z = 0, s= ' $%0.d e_2 = (0, %0.d, 0)$'% (b, b), size = 15)
    #     ax.text(x = 0, y = 0, z = c, s= ' $%0.d e_3 = (0, 0, %0.d)$' %(c, c), size = 15)

    #################################Axis Setting######################################
    ax.grid()
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 15])
    ax.set_zlim([0, 15])

    ax.set_xlabel("x-axis", size=18)
    ax.set_ylabel("y-axis", size=18)
    ax.set_zlabel("z-axis", size=18)

    # ax.set_title('Vector $(%0.d, %0.d, %.0d)$ Visualization' %(a, b, c), size = 20)

    ax.view_init(elev=20.0, azim=15)
