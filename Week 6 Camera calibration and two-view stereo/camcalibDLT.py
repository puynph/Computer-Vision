import numpy as np


def camcalibDLT(x_world, x_im):
    """
    :param x_world: World coordinatesm with shape (point_id, coordinates)
    :param x_im: Image coordinates with shape (point_id, coordinates)
    :return P: Camera projection matrix with shape (3,4)
    """
    # Create the matrix A 
    ##-your-code-starts-here-##
    A = []

    for i in range(len(x_world)):
        upper = [np.zeros(x_world[i].shape), x_world[i], -x_im[i, 1]*x_world[i]]
        upper = np.hstack(upper)
        lower = [x_world[i], np.zeros(x_world[i].shape), -x_im[i, 0]*x_world[i]]
        lower = np.hstack(lower)
        A.append(upper)
        A.append(lower)
    
    A = np.array(A)

        
    ##-your-code-ends-here-##
    
    # Perform homogeneous least squares fitting.
    # The best solution is given by the eigenvector of
    # A.T*A with the smallest eigenvalue.
    ##-your-code-starts-here-##
    A_ = np.matmul(A.T, A)
    eigenvalues, ev = np.linalg.eig(A_)
    idx = np.argmin(eigenvalues)
    P = ev[:,idx]
    ##-your-code-ends-here-##
    
    # Reshape the eigenvector into a projection matrix P
    P = np.reshape(P, (3, 4))  # here ev is the eigenvector from above
    #P = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=float)  # remove this and uncomment the line above
    
    return P