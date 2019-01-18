from osgeo import gdal, osr
import numpy as np
from sys import exit

def make_2D_grid(sitecoords, proj, maxrange, maxalt, horiz_res, vert_res):
    """Generate Cartesian coordinates for a regular 3-D grid based on
    radar specs.

    .. versionchanged:: 0.6.0
       using osr objects instead of PROJ.4 strings as parameter

    Parameters
    ----------
    sitecoords
    proj

        .. versionadded:: 0.6.0
           using osr objects instead of PROJ.4 strings as parameter

    maxrange
    maxalt
    horiz_res
    vert_res

    Returns
    -------
    output : float array of shape (num grid points, 3), a tuple of
        3 representing the grid shape

    """
    center = reproject(sitecoords[0], sitecoords[1],
                              projection_target=proj)
    # minz = sitecoords[2]
    llx = center[0] - maxrange
    lly = center[1] - maxrange
    x = np.arange(llx, llx + 2 * maxrange + horiz_res, horiz_res)
    y = np.arange(lly, lly + 2 * maxrange + horiz_res, horiz_res)
    z = np.arange(0., maxalt + vert_res, vert_res)
    xyz = gridaspoints(z, y, x)
    shape = (len(z), len(y), len(x))
    return xyz, shape

def reproject(*args, **kwargs):
    """Transform coordinates from a source projection to a target projection.

    Call signatures::

        reproject(C, **kwargs)
        reproject(X, Y, **kwargs)
        reproject(X, Y, Z, **kwargs)

    *C* is the np array of source coordinates.
    *X*, *Y* and *Z* specify arrays of x, y, and z coordinate values

    Parameters
    ----------
    C : multidimensional :class:`numpy:numpy.ndarray`
        Array of shape (...,2) or (...,3) with coordinates (x,y) or (x,y,z)
        respectively
    X : :class:`numpy:numpy.ndarray`
        Array of x coordinates
    Y : :class:`numpy:numpy.ndarray`
        Array of y coordinates
    Z : :class:`numpy:numpy.ndarray`
        Array of z coordinates

    Keyword Arguments
    -----------------
    projection_source : osr object
        defaults to EPSG(4326)
    projection_target : osr object
        defaults to EPSG(4326)

    Returns
    -------
    trans : :class:`numpy:numpy.ndarray`
        Array of reprojected coordinates x,y (...,2) or x,y,z (...,3)
        depending on input array.
    X, Y : :class:`numpy:numpy.ndarray`
        Arrays of reprojected x,y coordinates, shape depending on input array
    X, Y, Z: :class:`numpy:numpy.ndarray`
        Arrays of reprojected x,y,z coordinates, shape depending on input array

    Examples
    --------

    See :ref:`notebooks/georeferencing/wradlib_georef_example.ipynb`.

    """
    if len(args) == 1:
        C = np.asanyarray(args[0])
        cshape = C.shape
        numCols = C.shape[-1]
        C = C.reshape(-1, numCols)
        if numCols < 2 or numCols > 3:
            raise TypeError('Input Array column mismatch '
                            'to %s' % ('reproject'))
    else:
        if len(args) == 2:
            X, Y = (np.asanyarray(arg) for arg in args)
            numCols = 2
        elif len(args) == 3:
            X, Y, Z = (np.asanyarray(arg) for arg in args)
            zshape = Z.shape
            numCols = 3
        else:
            raise TypeError('Illegal arguments to %s' % ('reproject'))

        xshape = X.shape
        yshape = Y.shape

        if xshape != yshape:
            raise TypeError('Incompatible X, Y inputs to %s' % ('reproject'))

        if 'Z' in locals():
            if xshape != zshape:
                raise TypeError('Incompatible Z input to %s' % ('reproject'))
            C = np.concatenate([X.ravel()[:, None],
                                Y.ravel()[:, None],
                                Z.ravel()[:, None]], axis=1)
        else:
            C = np.concatenate([X.ravel()[:, None],
                                Y.ravel()[:, None]], axis=1)

    projection_source = kwargs.get('projection_source',
                                   get_default_projection())
    projection_target = kwargs.get('projection_target',
                                   get_default_projection())

    ct = osr.CoordinateTransformation(projection_source, projection_target)
    trans = np.array(ct.TransformPoints(C))

    if len(args) == 1:
        # here we could do this one
        # return(np.array(ct.TransformPoints(C))[...,0:numCols]))
        # or this one
        trans = trans[:, 0:numCols].reshape(cshape)
        return trans
    else:
        X = trans[:, 0].reshape(xshape)
        Y = trans[:, 1].reshape(yshape)
        if len(args) == 2:
            return X, Y
        if len(args) == 3:
            Z = trans[:, 2].reshape(zshape)
            return X, Y, Z

def get_default_projection():
    """Create a default projection object (wgs84)"""
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(4326)
    return proj

def gridaspoints(*arrs):
    """Creates an N-dimensional grid form arrs and returns grid points sequence
    of point coordinate pairs
    """
    # there is a small gotcha here.
    # with the convention following the 2013-08-30 sprint in Potsdam it was
    # agreed upon that arrays should have shapes (...,z,y,x) similar to the
    # convention that polar data should be (...,time,scan,azimuth,range)
    #
    # Still coordinate tuples are given in the order (x,y,z) [and hopefully not
    # more dimensions]. Therefore np.meshgrid must be fed the axis coordinates
    # in shape order (z,y,x) and the result needs to be reversed in order
    # for everything to work out.
    grid = tuple([dim.ravel()
                  for dim in reversed(np.meshgrid(*arrs, indexing='ij'))])
    return np.vstack(grid).transpose()