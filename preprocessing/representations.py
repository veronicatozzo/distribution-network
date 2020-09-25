import numpy as np 


def min_max_values(X):
    """Return min_max values across all dimensions of the vectors in X
    
    Parameters
    ----------
    X: list 
        X is a list of length N, each element of the list is of type
        ndarray with dimensions (n_i, d). 

    Returns
    -------
    list of floats: length 2d
        For each dimension the max and min value across 
        that dimension is returned.
    
    """
    outputs_values = []
    
    if len(X) == 0 :
        raise ValueError('No element in list of arrays.')

    for d in X[0].shape[1]:
        max_, min_ = np.max(X[0][:,d]), np.min(X[0][:, d])
        for r in X[1:]:
            max_, min_ = max(max_, np.max(r[:,d])), min(min_, np.min(r[:, d]))
        outputs_values.append((max_, min_))
    return outputs_values


def get_PDF_view(X, min_max_values, n_lines=25):
    """
    Returns the PDF view of each array in x. 

    Parameters
    ----------
    x: list or array-like
        The N matrices to represent in PDF format. 
        Each matrix has shape (n_i, d).

    min_max_values: list
        For each dimension of the matrices in x,
        a min and a max values shoulw be provided. 
        See function mix_max_values. 


    Returns
    -------
    array-like: shape=(N, n_lines, n_lines, ...) 
        The PDF representation of the input matrices.
        The times of n_lines depends on d.  
    """

    #TODO CHECKS
    # check that min-max-values has exatly d couples as the dimension in x 
    
    values = []
    lines = []
    for d in range(len(min_max_values)):
        max_, min_ = min_max_values[d]
        grid_lines = np.array([min_-1] + list(np.linspace(min_, max_, n_lines-1)) + [max_])
        lines.append(grid_lines)
        means = [np.mean([grid_lines[i], grid_lines[i+1]]) for i in range(n_lines)]
        x_ = np.array([means for i in range(n_lines)])
        x_ = x_ - np.mean(x_, axis=1)
        x_ /= np.max(np.abs(x_))
        values.append(x_)

    if isinstance(X, np.ndarray):
        X = [X]

    # TODO  (make it multi-dimensional?)
    final_res = []
    for ix in range(len(X)):
        x = X[ix]
        
        histogram = []
        for i in range(n_lines):
            to_consider = x[np.where(np.logical_and(x[:,0]>=lines[0][i], x[:,0]<lines[0][i+1]))[0], :]
            temp_ = []
            for j in range(n_lines):
                aux  = to_consider[np.where(np.logical_and(to_consider[:,1]>=lines[1][j], to_consider[:,1]<lines[1][j+1]))[0], :]  
                counts = aux.shape[0]   
                temp_.append(counts)
            histogram.append(temp_[::-1])
        histogram = np.array(histogram)
        histogram = histogram/np.max(histogram)
        res = np.array([histogram] + values)
        res = np.moveaxis(res, 0, -1)
        final_res.append(res)
    return np.array(final_res)


def get_CDF_view(X, min_max_values, n_lines=25):
    """
    Returns the CDF view of each array in x. 

    Parameters
    ----------
    x: list or array-like
        The N matrices to represent in PDF format. 
        Each matrix has shape (n_i, d).

    min_max_values: list
        For each dimension of the matrices in x,
        a min and a max values shoulw be provided. 
        See function mix_max_values. 


    Returns
    -------
    array-like: shape=(N, n_lines, n_lines, ...) 
        The CDF representation of the input matrices.
        The times of n_lines depends on d.  
    """

    #TODO CHECKS
    # check that min-max-values has exatly d couples as the dimension in x 
    
    pdfs = get_PDF_view(X, min_max_values, n_lines)
    cdfs = []
    for i in range(pdfs.shape[0]):
        p = pdfs[i]
        aux = np.zeros((n_lines, n_lines))
        for i in range(n_lines):
            for j in range(n_lines):
                aux[n_lines-1-i,j] = np.sum(p[:i, :j, 0])
        cdf = np.array([aux/np.max(aux), p[:, :, 1], p[:, :, 2]])
        cdf = np.moveaxis(cdf, 0, -1)
        cdfs.append(cdf)
    return cdfs