def get_residual(data, xaxis, yaxis, gamma, w0, interp_kind, return_background):
    """
    Convolve the input data with a Gaussian kernel whose standard deviation
    is a function of distance from the origin, and return the residual after
    subtracting this blurred map from the original.
    This is achieved not by varying the kernel size over the map, but instead
    by stretching the map with the inverse scale. This idea taken from:
    https://stackoverflow.com/questions/18624005/how-do-i-perform-a-convolution-in-python-with-a-variable-width-gaussian

    The reasoning behind it: We want to highlight substructure whose spatial
    scales themselves scale with radius. Using a fixed kernel size is too big
    in the inner disk (over-blurs) and too small in the outer disk (highlights
    noise).

    Args:
        data (2d array): Input data map that you wish to put through the filter.
        xaxis (1d arrray): x-axis on which the input data lies. Note the function
            assumes the axes are centered on the origin.
        yaxis (1d arrray): y-axis on which the input data lies. Note the function
            assumes the axes are centered on the origin.
        gamma (float): The radial power law index that you wish the kernel
            width to follow. For 0 > gamma > 1, the Gaussian kernel will
            increase as a function of radius.
        w0 (int): The width of the Gaussian kernel in units of original grid
            pixels.
        interp_kind (str): kwarg of scipy.interpolate, for example 'cubic'.
        return_background (bool): If True, return the blurred map instead of the
            residual map.
    """

    from scipy.ndimage.filters import gaussian_filter
    from scipy import interpolate
    import numpy as np

    input_data = data.copy()
    X, Y = xaxis.copy(), yaxis.copy()

    # Define the radial grid
    R = (X**2 + Y**2)**0.5

    # Stretch the original grid by desired radial stretch
    x = X * (R**(gamma))
    y = Y * (R**(gamma))

    # Prepare interp function of input data on original grid
    func1 = interpolate.interp2d(X, Y, input_data, kind=interp_kind)

    # Interpolate input data onto radially stretched grid
    stretched_data = func1(x, y)

    # Perform convolution on warped data and subtract result to get residual
    background = gaussian_filter(stretched_data, sigma=w0)
    highpass_residual = stretched_data - background

    if return_background:
        # Prepare interp function of background on radially stretched grid
        func2 = interpolate.interp2d(x, y, background, kind=interp_kind)

        # Interpolate residual data back onto the original grid
        output_data = func2(X, Y)

        return output_data

    else:
        # Prepare interp function of residual data on radially stretched grid
        func2 = interpolate.interp2d(x, y, highpass_residual, kind=interp_kind)

        # Interpolate residual data back onto the original grid
        output_data = func2(X, Y)

        return output_data
