import numpy as np
from pyquaternion import Quaternion


def rotate_mag2nec(quaternions_mag2nec, mag_data):
    """
    Rotate MAG data from MAG frame to NEC frame using quaternions.
    Args:
        quaternions_mag2nec: Quaternions to rotate MAG data from MAG frame to NEC frame.
        mag_data: MAG data to rotate.
    Returns:
        Rotated MAG data in NEC frame.
    """
    # Check if the input quaternions and MAG data are of the same length.
    if len(quaternions_mag2nec) != len(mag_data):
        raise ValueError("Length of quaternions and MAG data should be the same.")
    # Convert the quaternions to pyquaternion format.
    py_quaternions = [Quaternion(w, x, y, z) for x, y, z, w in quaternions_mag2nec]
    # Rotate the MAG data from MAG frame to NEC frame using the quaternions.
    nec_data = np.array([q.rotate(vec) for q, vec in zip(py_quaternions, mag_data)])
    return nec_data


def rotate_nec2mag(quaternions_fgm2nec, nec_data):
    """
    Rotate NEC data from NEC frame to MAG frame using quaternions.
    Args:
        quaternions_fgm2nec: Quaternions to rotate NEC data from NEC frame to MAG frame.
        nec_data: NEC data to rotate.
    Returns:
        Rotated NEC data in MAG frame.
    """
    # Check if the input quaternions and NEC data are of the same length.
    if len(quaternions_fgm2nec) != len(nec_data):
        raise ValueError("Length of quaternions and NEC data should be the same.")
    # Convert the quaternions to pyquaternion format and invert the fgm2nec quaternions to nec2fgm quaternions.
    py_quaternions_inversed = [Quaternion(w, x, y, z).inverse for x, y, z, w in quaternions_fgm2nec]
    # Rotate the NEC data from NEC frame to MAG frame using the inverted quaternions.
    mag_data = np.array([q.rotate(vec) for q, vec in zip(py_quaternions_inversed, nec_data)])
    # Return the rotated NEC data in MAG frame.
    return mag_data
