import numpy as np
from .morphology import generate_binary_structure
try:
    import ni_label
except:
    try:
        from backend.custom_scipy.src import ni_label
    except:
        print('LOADING NI_LABEL FAILED')

def label(input, structure=None, output=None):
    input = np.asarray(input)
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    if structure is None:
        structure = generate_binary_structure(input.ndim, 1)
    structure = np.asarray(structure, dtype=bool)
    if structure.ndim != input.ndim:
        raise RuntimeError('structure and input must have equal rank')
    for ii in structure.shape:
        if ii != 3:
            raise ValueError('structure dimensions must be equal to 3')

    # Use 32 bits if it's large enough for this image.
    # _ni_label.label() needs two entries for background and
    # foreground tracking
    need_64bits = input.size >= (2**31 - 2)

    if isinstance(output, np.ndarray):
        if output.shape != input.shape:
            raise ValueError("output shape not correct")
        caller_provided_output = True
    else:
        caller_provided_output = False
        if output is None:
            output = np.empty(input.shape, np.intp if need_64bits else np.int32)
        else:
            output = np.empty(input.shape, output)

    # handle scalars, 0-D arrays
    if input.ndim == 0 or input.size == 0:
        if input.ndim == 0:
            # scalar
            maxlabel = 1 if (input != 0) else 0
            output[...] = maxlabel
        else:
            # 0-D
            maxlabel = 0
        if caller_provided_output:
            return maxlabel
        else:
            return output, maxlabel

    try:
        max_label = ni_label._label(input, structure, output)
    except ni_label.NeedMoreBits as e:
        # Make another attempt with enough bits, then try to cast to the
        # new type.
        tmp_output = np.empty(input.shape, np.intp if need_64bits else np.int32)
        max_label = ni_label._label(input, structure, tmp_output)
        output[...] = tmp_output[...]
        if not np.all(output == tmp_output):
            # refuse to return bad results
            raise RuntimeError(
                "insufficient bit-depth in requested output type"
            ) from e

    if caller_provided_output:
        # result was written in-place
        return max_label
    else:
        return output, max_label