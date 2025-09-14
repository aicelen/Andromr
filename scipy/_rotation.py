from __future__ import annotations

from collections.abc import Iterable, Iterator
from types import EllipsisType, ModuleType, NotImplementedType

import numpy as np

import scipy.spatial.transform._rotation_cy as cython_backend
import scipy.spatial.transform._rotation_xp as xp_backend
from scipy.spatial.transform._rotation_groups import create_group
from scipy._lib._array_api import (
    array_namespace,
    Array,
    is_numpy,
    ArrayLike,
    is_lazy_array,
    xp_capabilities,
    xp_promote,
)
import scipy._lib.array_api_extra as xpx

backend_registry = {array_namespace(np.empty(0)): cython_backend}




class Rotation:
    """Rotation in 3 dimensions.

    This class provides an interface to initialize from and represent rotations
    with:

    - Quaternions
    - Rotation Matrices
    - Rotation Vectors
    - Modified Rodrigues Parameters
    - Euler Angles
    - Davenport Angles (Generalized Euler Angles)

    The following operations on rotations are supported:

    - Application on vectors
    - Rotation Composition
    - Rotation Inversion
    - Rotation Indexing

    Indexing within a rotation is supported since multiple rotation transforms
    can be stored within a single `Rotation` instance.

    To create `Rotation` objects use ``from_...`` methods (see examples below).
    ``Rotation(...)`` is not supposed to be instantiated directly.

    Attributes
    ----------
    single

    Methods
    -------
    __len__
    from_quat
    from_matrix
    from_rotvec
    from_mrp
    from_euler
    from_davenport
    as_quat
    as_matrix
    as_rotvec
    as_mrp
    as_euler
    as_davenport
    concatenate
    apply
    __mul__
    __pow__
    inv
    magnitude
    approx_equal
    mean
    reduce
    create_group
    __getitem__
    identity
    random
    align_vectors

    See Also
    --------
    Slerp

    Notes
    -----
    .. versionadded:: 1.2.0

    Examples
    --------
    >>> from scipy.spatial.transform import Rotation as R
    >>> import numpy as np

    A `Rotation` instance can be initialized in any of the above formats and
    converted to any of the others. The underlying object is independent of the
    representation used for initialization.

    Consider a counter-clockwise rotation of 90 degrees about the z-axis. This
    corresponds to the following quaternion (in scalar-last format):

    >>> r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])

    The rotation can be expressed in any of the other formats:

    >>> r.as_matrix()
    array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
    [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    >>> r.as_rotvec()
    array([0.        , 0.        , 1.57079633])
    >>> r.as_euler('zyx', degrees=True)
    array([90.,  0.,  0.])

    The same rotation can be initialized using a rotation matrix:

    >>> r = R.from_matrix([[0, -1, 0],
    ...                    [1, 0, 0],
    ...                    [0, 0, 1]])

    Representation in other formats:

    >>> r.as_quat()
    array([0.        , 0.        , 0.70710678, 0.70710678])
    >>> r.as_rotvec()
    array([0.        , 0.        , 1.57079633])
    >>> r.as_euler('zyx', degrees=True)
    array([90.,  0.,  0.])

    The rotation vector corresponding to this rotation is given by:

    >>> r = R.from_rotvec(np.pi/2 * np.array([0, 0, 1]))

    Representation in other formats:

    >>> r.as_quat()
    array([0.        , 0.        , 0.70710678, 0.70710678])
    >>> r.as_matrix()
    array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
           [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    >>> r.as_euler('zyx', degrees=True)
    array([90.,  0.,  0.])

    The ``from_euler`` method is quite flexible in the range of input formats
    it supports. Here we initialize a single rotation about a single axis:

    >>> r = R.from_euler('z', 90, degrees=True)

    Again, the object is representation independent and can be converted to any
    other format:

    >>> r.as_quat()
    array([0.        , 0.        , 0.70710678, 0.70710678])
    >>> r.as_matrix()
    array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
           [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    >>> r.as_rotvec()
    array([0.        , 0.        , 1.57079633])

    It is also possible to initialize multiple rotations in a single instance
    using any of the ``from_...`` functions. Here we initialize a stack of 3
    rotations using the ``from_euler`` method:

    >>> r = R.from_euler('zyx', [
    ... [90, 0, 0],
    ... [0, 45, 0],
    ... [45, 60, 30]], degrees=True)

    The other representations also now return a stack of 3 rotations. For
    example:

    >>> r.as_quat()
    array([[0.        , 0.        , 0.70710678, 0.70710678],
           [0.        , 0.38268343, 0.        , 0.92387953],
           [0.39190384, 0.36042341, 0.43967974, 0.72331741]])

    Applying the above rotations onto a vector:

    >>> v = [1, 2, 3]
    >>> r.apply(v)
    array([[-2.        ,  1.        ,  3.        ],
           [ 2.82842712,  2.        ,  1.41421356],
           [ 2.24452282,  0.78093109,  2.89002836]])

    A `Rotation` instance can be indexed and sliced as if it were a single
    1D array or list:

    >>> r.as_quat()
    array([[0.        , 0.        , 0.70710678, 0.70710678],
           [0.        , 0.38268343, 0.        , 0.92387953],
           [0.39190384, 0.36042341, 0.43967974, 0.72331741]])
    >>> p = r[0]
    >>> p.as_matrix()
    array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
           [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    >>> q = r[1:3]
    >>> q.as_quat()
    array([[0.        , 0.38268343, 0.        , 0.92387953],
           [0.39190384, 0.36042341, 0.43967974, 0.72331741]])

    In fact it can be converted to numpy.array:

    >>> r_array = np.asarray(r)
    >>> r_array.shape
    (3,)
    >>> r_array[0].as_matrix()
    array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
           [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    Multiple rotations can be composed using the ``*`` operator:

    >>> r1 = R.from_euler('z', 90, degrees=True)
    >>> r2 = R.from_rotvec([np.pi/4, 0, 0])
    >>> v = [1, 2, 3]
    >>> r2.apply(r1.apply(v))
    array([-2.        , -1.41421356,  2.82842712])
    >>> r3 = r2 * r1 # Note the order
    >>> r3.apply(v)
    array([-2.        , -1.41421356,  2.82842712])

    A rotation can be composed with itself using the ``**`` operator:

    >>> p = R.from_rotvec([1, 0, 0])
    >>> q = p ** 2
    >>> q.as_rotvec()
    array([2., 0., 0.])

    Finally, it is also possible to invert rotations:

    >>> r1 = R.from_euler('z', [90, 45], degrees=True)
    >>> r2 = r1.inv()
    >>> r2.as_euler('zyx', degrees=True)
    array([[-90.,   0.,   0.],
           [-45.,   0.,   0.]])

    The following function can be used to plot rotations with Matplotlib by
    showing how they transform the standard x, y, z coordinate axes:

    >>> import matplotlib.pyplot as plt

    >>> def plot_rotated_axes(ax, r, name=None, offset=(0, 0, 0), scale=1):
    ...     colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    ...     loc = np.array([offset, offset])
    ...     for i, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
    ...                                       colors)):
    ...         axlabel = axis.axis_name
    ...         axis.set_label_text(axlabel)
    ...         axis.label.set_color(c)
    ...         axis.line.set_color(c)
    ...         axis.set_tick_params(colors=c)
    ...         line = np.zeros((2, 3))
    ...         line[1, i] = scale
    ...         line_rot = r.apply(line)
    ...         line_plot = line_rot + loc
    ...         ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)
    ...         text_loc = line[1]*1.2
    ...         text_loc_rot = r.apply(text_loc)
    ...         text_plot = text_loc_rot + loc[0]
    ...         ax.text(*text_plot, axlabel.upper(), color=c,
    ...                 va="center", ha="center")
    ...     ax.text(*offset, name, color="k", va="center", ha="center",
    ...             bbox={"fc": "w", "alpha": 0.8, "boxstyle": "circle"})

    Create three rotations - the identity and two Euler rotations using
    intrinsic and extrinsic conventions:

    >>> r0 = R.identity()
    >>> r1 = R.from_euler("ZYX", [90, -30, 0], degrees=True)  # intrinsic
    >>> r2 = R.from_euler("zyx", [90, -30, 0], degrees=True)  # extrinsic

    Add all three rotations to a single plot:

    >>> ax = plt.figure().add_subplot(projection="3d", proj_type="ortho")
    >>> plot_rotated_axes(ax, r0, name="r0", offset=(0, 0, 0))
    >>> plot_rotated_axes(ax, r1, name="r1", offset=(3, 0, 0))
    >>> plot_rotated_axes(ax, r2, name="r2", offset=(6, 0, 0))
    >>> _ = ax.annotate(
    ...     "r0: Identity Rotation\\n"
    ...     "r1: Intrinsic Euler Rotation (ZYX)\\n"
    ...     "r2: Extrinsic Euler Rotation (zyx)",
    ...     xy=(0.6, 0.7), xycoords="axes fraction", ha="left"
    ... )
    >>> ax.set(xlim=(-1.25, 7.25), ylim=(-1.25, 1.25), zlim=(-1.25, 1.25))
    >>> ax.set(xticks=range(-1, 8), yticks=[-1, 0, 1], zticks=[-1, 0, 1])
    >>> ax.set_aspect("equal", adjustable="box")
    >>> ax.figure.set_size_inches(6, 5)
    >>> plt.tight_layout()

    Show the plot:

    >>> plt.show()

    These examples serve as an overview into the `Rotation` class and highlight
    major functionalities. For more thorough examples of the range of input and
    output formats supported, consult the individual method's examples.

    """

    def __init__(
        self,
        quat: ArrayLike,
        normalize: bool = True,
        copy: bool = True,
        scalar_first: bool = False,
    ):
        xp = array_namespace(quat)
        self._xp = xp
        quat = _promote(quat, xp=xp)
        if quat.shape[-1] != 4:
            raise ValueError(
                f"Expected `quat` to have shape (..., 4), got {quat.shape}."
            )
        # Single NumPy quats or list of quats are accelerated by the cython backend.
        # This backend needs inputs with fixed ndim, so we always expand to 2D and
        # select the 0th element if quat was single to get the correct shape. For other
        # frameworks and quaternion tensors we use the generic array API backend.
        self._single = quat.ndim == 1 and is_numpy(xp)
        if self._single:
            quat = xpx.atleast_nd(quat, ndim=2, xp=xp)
        self._backend = _select_backend(xp, cython_compatible=quat.ndim < 3)
        self._quat: Array = self._backend.from_quat(
            quat, normalize=normalize, copy=copy, scalar_first=scalar_first
        )

    @staticmethod
    @xp_capabilities(
        skip_backends=[("dask.array", "missing linalg.cross/det functions")]
    )
    def from_quat(quat: ArrayLike, *, scalar_first: bool = False) -> Rotation:
        """Initialize from quaternions.

        Rotations in 3 dimensions can be represented using unit norm
        quaternions [1]_.

        The 4 components of a quaternion are divided into a scalar part ``w``
        and a vector part ``(x, y, z)`` and can be expressed from the angle
        ``theta`` and the axis ``n`` of a rotation as follows::

            w = cos(theta / 2)
            x = sin(theta / 2) * n_x
            y = sin(theta / 2) * n_y
            z = sin(theta / 2) * n_z

        There are 2 conventions to order the components in a quaternion:

        - scalar-first order -- ``(w, x, y, z)``
        - scalar-last order -- ``(x, y, z, w)``

        The choice is controlled by `scalar_first` argument.
        By default, it is False and the scalar-last order is assumed.

        Advanced users may be interested in the "double cover" of 3D space by
        the quaternion representation [2]_. As of version 1.11.0, the
        following subset (and only this subset) of operations on a `Rotation`
        ``r`` corresponding to a quaternion ``q`` are guaranteed to preserve
        the double cover property: ``r = Rotation.from_quat(q)``,
        ``r.as_quat(canonical=False)``, ``r.inv()``, and composition using the
        ``*`` operator such as ``r*r``.

        Parameters
        ----------
        quat : array_like, shape (N, 4) or (4,)
            Each row is a (possibly non-unit norm) quaternion representing an
            active rotation. Each quaternion will be normalized to unit norm.
        scalar_first : bool, optional
            Whether the scalar component goes first or last.
            Default is False, i.e. the scalar-last order is assumed.

        Returns
        -------
        rotation : `Rotation` instance
            Object containing the rotations represented by input quaternions.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        .. [2] Hanson, Andrew J. "Visualizing quaternions."
            Morgan Kaufmann Publishers Inc., San Francisco, CA. 2006.

        Examples
        --------
        >>> from scipy.spatial.transform import Rotation as R

        A rotation can be initialzied from a quaternion with the scalar-last
        (default) or scalar-first component order as shown below:

        >>> r = R.from_quat([0, 0, 0, 1])
        >>> r.as_matrix()
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
        >>> r = R.from_quat([1, 0, 0, 0], scalar_first=True)
        >>> r.as_matrix()
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])

        It is possible to initialize multiple rotations in a single object by
        passing a 2-dimensional array:

        >>> r = R.from_quat([
        ... [1, 0, 0, 0],
        ... [0, 0, 0, 1]
        ... ])
        >>> r.as_quat()
        array([[1., 0., 0., 0.],
               [0., 0., 0., 1.]])
        >>> r.as_quat().shape
        (2, 4)

        It is also possible to have a stack of a single rotation:

        >>> r = R.from_quat([[0, 0, 0, 1]])
        >>> r.as_quat()
        array([[0., 0., 0., 1.]])
        >>> r.as_quat().shape
        (1, 4)

        Quaternions are normalized before initialization.

        >>> r = R.from_quat([0, 0, 1, 1])
        >>> r.as_quat()
        array([0.        , 0.        , 0.70710678, 0.70710678])
        """
        return Rotation(quat, normalize=True, scalar_first=scalar_first)

    @staticmethod
    @xp_capabilities(
        skip_backends=[("dask.array", "missing linalg.cross/det functions")]
    )
    def from_matrix(matrix: ArrayLike) -> Rotation:
        """Initialize from rotation matrix.

        Rotations in 3 dimensions can be represented with 3 x 3 orthogonal
        matrices [1]_. If the input is not orthogonal, an approximation is
        created by orthogonalizing the input matrix using the method described
        in [2]_, and then converting the orthogonal rotation matrices to
        quaternions using the algorithm described in [3]_. Matrices must be
        right-handed.

        Parameters
        ----------
        matrix : array_like, shape (N, 3, 3) or (3, 3)
            A single matrix or a stack of matrices, where ``matrix[i]`` is
            the i-th matrix.

        Returns
        -------
        rotation : `Rotation` instance
            Object containing the rotations represented by the rotation
            matrices.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        .. [2] https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        .. [3] F. Landis Markley, "Unit Quaternion from Rotation Matrix",
               Journal of guidance, control, and dynamics vol. 31.2, pp.
               440-442, 2008.

        Examples
        --------
        >>> from scipy.spatial.transform import Rotation as R
        >>> import numpy as np

        Initialize a single rotation:

        >>> r = R.from_matrix([
        ... [0, -1, 0],
        ... [1, 0, 0],
        ... [0, 0, 1]])
        >>> r.single
        True
        >>> r.as_matrix().shape
        (3, 3)

        Initialize multiple rotations in a single object:

        >>> r = R.from_matrix([
        ... [
        ...     [0, -1, 0],
        ...     [1, 0, 0],
        ...     [0, 0, 1],
        ... ],
        ... [
        ...     [1, 0, 0],
        ...     [0, 0, -1],
        ...     [0, 1, 0],
        ... ]])
        >>> r.as_matrix().shape
        (2, 3, 3)
        >>> r.single
        False
        >>> len(r)
        2

        If input matrices are not special orthogonal (orthogonal with
        determinant equal to +1), then a special orthogonal estimate is stored:

        >>> a = np.array([
        ... [0, -0.5, 0],
        ... [0.5, 0, 0],
        ... [0, 0, 0.5]])
        >>> np.linalg.det(a)
        0.125
        >>> r = R.from_matrix(a)
        >>> matrix = r.as_matrix()
        >>> matrix
        array([[ 0., -1.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  0.,  1.]])
        >>> np.linalg.det(matrix)
        1.0

        It is also possible to have a stack containing a single rotation:

        >>> r = R.from_matrix([[
        ... [0, -1, 0],
        ... [1, 0, 0],
        ... [0, 0, 1]]])
        >>> r.as_matrix()
        array([[[ 0., -1.,  0.],
                [ 1.,  0.,  0.],
                [ 0.,  0.,  1.]]])
        >>> r.as_matrix().shape
        (1, 3, 3)

        Notes
        -----
        This function was called from_dcm before.

        .. versionadded:: 1.4.0
        """
        xp = array_namespace(matrix)
        matrix = _promote(matrix, xp=xp)
        # Resulting quat will have 1 less dimension than matrix
        backend = _select_backend(xp, cython_compatible=matrix.ndim < 4)
        quat = backend.from_matrix(matrix)
        return Rotation._from_raw_quat(quat, xp=xp, backend=backend)

    @staticmethod
    @xp_capabilities(
        skip_backends=[("dask.array", "missing linalg.cross/det functions")]
    )
    def from_rotvec(rotvec: ArrayLike, degrees: bool = False) -> Rotation:
        """Initialize from rotation vectors.

        A rotation vector is a 3 dimensional vector which is co-directional to
        the axis of rotation and whose norm gives the angle of rotation [1]_.

        Parameters
        ----------
        rotvec : array_like, shape (N, 3) or (3,)
            A single vector or a stack of vectors, where `rot_vec[i]` gives
            the ith rotation vector.
        degrees : bool, optional
            If True, then the given magnitudes are assumed to be in degrees.
            Default is False.

            .. versionadded:: 1.7.0

        Returns
        -------
        rotation : `Rotation` instance
            Object containing the rotations represented by input rotation
            vectors.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotation_vector

        Examples
        --------
        >>> from scipy.spatial.transform import Rotation as R
        >>> import numpy as np

        Initialize a single rotation:

        >>> r = R.from_rotvec(np.pi/2 * np.array([0, 0, 1]))
        >>> r.as_rotvec()
        array([0.        , 0.        , 1.57079633])
        >>> r.as_rotvec().shape
        (3,)

        Initialize a rotation in degrees, and view it in degrees:

        >>> r = R.from_rotvec(45 * np.array([0, 1, 0]), degrees=True)
        >>> r.as_rotvec(degrees=True)
        array([ 0., 45.,  0.])

        Initialize multiple rotations in one object:

        >>> r = R.from_rotvec([
        ... [0, 0, np.pi/2],
        ... [np.pi/2, 0, 0]])
        >>> r.as_rotvec()
        array([[0.        , 0.        , 1.57079633],
               [1.57079633, 0.        , 0.        ]])
        >>> r.as_rotvec().shape
        (2, 3)

        It is also possible to have a stack of a single rotation:

        >>> r = R.from_rotvec([[0, 0, np.pi/2]])
        >>> r.as_rotvec().shape
        (1, 3)

        """
        xp = array_namespace(rotvec)
        rotvec = _promote(rotvec, xp=xp)
        backend = _select_backend(xp, cython_compatible=rotvec.ndim < 3)
        quat = backend.from_rotvec(rotvec, degrees=degrees)
        return Rotation._from_raw_quat(quat, xp=xp, backend=backend)

    @staticmethod
    @xp_capabilities(
        skip_backends=[("dask.array", "missing linalg.cross/det functions")]
    )
    def from_euler(seq: str, angles: ArrayLike, degrees: bool = False) -> Rotation:
        """Initialize from Euler angles.

        Rotations in 3-D can be represented by a sequence of 3
        rotations around a sequence of axes. In theory, any three axes spanning
        the 3-D Euclidean space are enough. In practice, the axes of rotation are
        chosen to be the basis vectors.

        The three rotations can either be in a global frame of reference
        (extrinsic) or in a body centred frame of reference (intrinsic), which
        is attached to, and moves with, the object under rotation [1]_.

        Parameters
        ----------
        seq : string
            Specifies sequence of axes for rotations. Up to 3 characters
            belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or
            {'x', 'y', 'z'} for extrinsic rotations. Extrinsic and intrinsic
            rotations cannot be mixed in one function call.
        angles : float or array_like, shape (N,) or (N, [1 or 2 or 3])
            Euler angles specified in radians (`degrees` is False) or degrees
            (`degrees` is True).
            For a single character `seq`, `angles` can be:

            - a single value
            - array_like with shape (N,), where each `angle[i]`
              corresponds to a single rotation
            - array_like with shape (N, 1), where each `angle[i, 0]`
              corresponds to a single rotation

            For 2- and 3-character wide `seq`, `angles` can be:

            - array_like with shape (W,) where `W` is the width of
              `seq`, which corresponds to a single rotation with `W` axes
            - array_like with shape (N, W) where each `angle[i]`
              corresponds to a sequence of Euler angles describing a single
              rotation

        degrees : bool, optional
            If True, then the given angles are assumed to be in degrees.
            Default is False.

        Returns
        -------
        rotation : `Rotation` instance
            Object containing the rotation represented by the sequence of
            rotations around given axes with given angles.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Euler_angles#Definition_by_intrinsic_rotations

        Examples
        --------
        >>> from scipy.spatial.transform import Rotation as R

        Initialize a single rotation along a single axis:

        >>> r = R.from_euler('x', 90, degrees=True)
        >>> r.as_quat().shape
        (4,)

        Initialize a single rotation with a given axis sequence:

        >>> r = R.from_euler('zyx', [90, 45, 30], degrees=True)
        >>> r.as_quat().shape
        (4,)

        Initialize a stack with a single rotation around a single axis:

        >>> r = R.from_euler('x', [90], degrees=True)
        >>> r.as_quat().shape
        (1, 4)

        Initialize a stack with a single rotation with an axis sequence:

        >>> r = R.from_euler('zyx', [[90, 45, 30]], degrees=True)
        >>> r.as_quat().shape
        (1, 4)

        Initialize multiple elementary rotations in one object:

        >>> r = R.from_euler('x', [90, 45, 30], degrees=True)
        >>> r.as_quat().shape
        (3, 4)

        Initialize multiple rotations in one object:

        >>> r = R.from_euler('zyx', [[90, 45, 30], [35, 45, 90]], degrees=True)
        >>> r.as_quat().shape
        (2, 4)

        """
        xp = array_namespace(angles)
        angles = _promote(angles, xp=xp)
        backend = _select_backend(xp, cython_compatible=angles.ndim < 3)
        quat = backend.from_euler(seq, angles, degrees=degrees)
        return Rotation._from_raw_quat(quat, xp=xp, backend=backend)

