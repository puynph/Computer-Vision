import numpy as np


def trianglin(P1, P2, x1, x2):
    """
    :param P1: Projection matrix for image 1 with shape (3,4)
    :param P2: Projection matrix for image 2 with shape (3,4)
    :param x1: Image coordinates for a point in image 1
    :param x2: Image coordinates for a point in image 2
    :return X: Triangulated world coordinates
    """
    # Form A and get the least squares solution from the eigenvector 
    # corresponding to the smallest eigenvalue
    ##-your-code-starts-here-##
    x1_cross_P1 = np.dot(np.array([[0, -1, x1[1]], 
                          [1, 0, -x1[0]], 
                          [-x1[1], x1[0], 0]]), P1)
    x2_cross_P2 = np.dot(np.array([[0, -1, x2[1]],
                          [1, 0, -x2[0]], 
                          [-x2[1], x2[0], 0]]), P2)
    
    # Stack the cross-product matrices vertically to form matrix A
    A = np.vstack([x1_cross_P1, x2_cross_P2])

    A_ = np.matmul(A.T, A)
    eigenvalues, ev = np.linalg.eig(A_)
    idx = np.argmin(eigenvalues)
    X_homogeneous = ev[:,idx]
    
    # Convert homogeneous coordinates to cartesian coordinates
    X = X_homogeneous[:-1] / X_homogeneous[-1]
    
    ##-your-code-ends-here-##
    
    return X

