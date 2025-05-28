#!/usr/bin/env python3
"""
-r^t * T = Coordinate of project/camera center:
r^t = inverse/transpose of the 3x3 rotation matrix composed from the quaternion
T = translation vector
python3 matrix.py output.csv >out.txt
combined filed parsing and matrix formation
command: python3 matrix.py images.txt camera.txt
parse_data.py will first parse images.txt and redirect to parsed_data.csv
matrix.py will then take the parsed_data.csv and redirect intrinsic and extrinsic
matrix to out_matrix.txt with the following info:
1. camera model
2. resolution
3. 1 intrinsic matrix
4. extrinsic matrix for each image w/ image name
"""

import sys
import csv
import math
import numpy as np
from . import image_position_extractor
import json
import os


# https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)

    Parameters
    ----------
    x, y, z, w : float
        A 4 element array representing the quaternion (x,y,z,w)

    Returns
    -------
    roll: float
        rotation around x in radians (counterclockwise)
    pitch: float
        rotation around y in radians (counterclockwise)
    yaw: float
        rotation around z in radians (counterclockwise)
    """

    # roll (x-axis rotation)
    sinr_cosp = +2.0 * (w * x + y * z)
    cosr_cosp = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = +2.0 * (w * y - z * x)
    sinp = +1.0 if sinp > +1.0 else sinp
    sinp = -1.0 if sinp < -1.0 else sinp
    pitch_y = math.asin(sinp)

    # yaw (x-axis rotation)
    siny_cosp = +2.0 * (w * z + x * y)
    cosy_cosp = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(siny_cosp, cosy_cosp)

    return roll_x, pitch_y, yaw_z  # in radians


# https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
def quaternion_rotation_matrix(qw, qx, qy, qz) -> np.ndarray:
    # w   x   y   z
    """
    Convert a quaternion into a full three-dimensional rotation matrix.

    Parameters
    ----------
    qw, qx, qy, qz : float
        A 4 element array representing the quaternion (qw,qx,qy,qz)

    Returns
    -------
    rotation_matrix : np.ndarray
        A 3x3 element matrix representing the full 3D rotation matrix.
        This rotation matrix converts a point in the local reference
        frame to a point in the global reference frame.
    """

    # First row of the rotation matrix
    r00 = 1 - 2 * qy**2 - 2 * qz**2
    r01 = 2 * qx * qy - 2 * qz * qw
    r02 = 2 * qx * qz + 2 * qy * qw

    # Second row of the rotation matrix
    r10 = 2 * qx * qy + 2 * qz * qw
    r11 = 1 - 2 * qx**2 - 2 * qz**2
    r12 = 2 * qy * qz - 2 * qx * qw

    # Third row of the rotation matrix
    r20 = 2 * qx * qz - 2 * qy * qw
    r21 = 2 * qy * qz + 2 * qx * qw
    r22 = 1 - 2 * qx**2 - 2 * qy**2

    # 3x3 rotation matrix
    rotation_matrix = np.array([[r00, r01, r02],
                                [r10, r11, r12],
                                [r20, r21, r22]])
    # np.set_printoptions(threshold=sys.maxsize)
    return rotation_matrix

# Function from https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
# Authored by Peter
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        raise ValueError("Input vectors must not be zero vectors.")

    a = (vec1 / norm_vec1).reshape(3)
    b = (vec2 / norm_vec2).reshape(3)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    # Check for collinearity
    if s == 0: # Vectors are collinear
        if c > 0.99999: # Parallel (c is close to 1)
            return np.eye(3)
        elif c < -0.99999: # Anti-parallel (c is close to -1)
            # Return a 180 degree rotation matrix.
            # One way: find an arbitrary axis orthogonal to 'a' and rotate by 180 deg.
            # Simpler: R = 2 * np.outer(a, a) - np.eye(3) if a is unit.
            # Or, if a = (1,0,0), b = (-1,0,0), rot around Y: [[-1,0,0],[0,1,0],[0,0,-1]]
            # A general approach for 180 deg rotation:
            # If a_x, a_y, a_z are components of vector a:
            # R = [[2*a_x**2-1, 2*a_x*a_y,   2*a_x*a_z],
            #      [2*a_x*a_y,   2*a_y**2-1, 2*a_y*a_z],
            #      [2*a_x*a_z,   2*a_y*a_z,   2*a_z**2-1]]
            # This is equivalent to np.outer(a,a)*2 - np.eye(3)
            # However, this gives reflection if not careful.
            # For 180 deg rotation: if a=(1,0,0), b=(-1,0,0), should be like diag([-1, -1, 1]) or diag([-1,1,-1]) etc.
            # A robust way to find a 180-degree rotation for anti-parallel vectors:
            # If a = [1,0,0], b = [-1,0,0], a rotation could be around Z by 180 deg (diag([-1,-1,1]))
            # or around Y by 180 deg (diag([-1,1,-1])).
            # The original formula for axis-angle R = I + sin(theta)*K + (1-cos(theta))*K^2.
            # For theta=pi (180 deg), sin(theta)=0, cos(theta)=-1. R = I + 2*K^2.
            # K is the skew-symmetric matrix of an axis u. This axis u needs to be found.
            # Any axis perpendicular to 'a' will do.
            # Let's try to find a perpendicular vector to 'a'.
            if np.allclose(a, [1, 0, 0]) or np.allclose(a, [-1, 0, 0]): # if a is along x-axis
                u = np.array([0, 1, 0]) # Rotate around y-axis
            else:
                u = np.cross(a, [1, 0, 0]) # Try to find a perpendicular axis
                if np.linalg.norm(u) < 1e-6: # If a was parallel to [1,0,0], u would be zero
                    u = np.cross(a, [0, 1, 0]) # Try another one
            u = u / np.linalg.norm(u)
            # K matrix for this axis u
            k_u = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
            return np.eye(3) + 2 * k_u.dot(k_u)
        else: # Should not happen if s=0 implies c is 1 or -1 for unit vectors
            return np.eye(3)


    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    # Using s**2 (s squared) ensures we use a positive value in the denominator
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix

def get_extrinsic(center_point, fp: str = "parsed_data.csv"):

    
    # contrains filepath and extrinsic matrix
    filepaths = []
    extrinsic_matrices = []
    with open(fp) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        row = next(csv_reader)  # start with second line

        for row in csv_reader:
            image_name = str(row[0])
            filepaths.append(image_name)

            qw = float(row[1])
            qx = float(row[2])
            qy = float(row[3])
            qz = float(row[4])

            tx = float(row[5])
            ty = float(row[6])
            tz = float(row[7])

            # find extrinsic matrix
            T = np.array([tx, ty, tz])
            r = quaternion_rotation_matrix(qw, qx, qy, qz)  # rotational matrix


            extrinsic = np.zeros((4,4))
            extrinsic[0:3,0:3] = r
            extrinsic[0:3,3] = T

            #T is not the position of the camera
            #r_t = r.transpose()
            #camera = -r_t @T
            extrinsic[3][3] = 1
            c2w = np.linalg.inv(extrinsic)

            # convert from OPENCV to OPENGL coordinates
            conversion = np.array([[1,0,0,0],
                                   [0,-1,0,0],
                                   [0,0,-1,0],
                                    [0,0,0,1]])
            # flips y and z coords
            c2w =  c2w @ conversion

            extrinsic_matrices.append(c2w)

            # Center extrinsics around center point:
            #extrinsic[0:3,3] += center_point

    # stack all extrinsic to perform faster transformations to the whole stack
    extrinsic_matrices = np.stack(extrinsic_matrices,axis=0)
    print(extrinsic_matrices.shape)
    avg_y_axis = np.sum(extrinsic_matrices[:,0:3,1], axis=0)
    avg_y_axis = avg_y_axis/np.linalg.norm(avg_y_axis)
    print("Consensus Y axis: ",avg_y_axis)

    # Find a matrix to rotate the average y axis with the y-axis unit vector thus aligning every extrinsic to point in the same direction
    Rot = np.zeros((4,4))
    Rot[0:3,0:3] = rotation_matrix_from_vectors(avg_y_axis,np.asarray([0,0,1]))
    Rot[-1,-1] = 1
    Rot = np.expand_dims(Rot,axis=0)

    # Rotate Extrinsic to all face up
    extrinsic_matrices = Rot @ extrinsic_matrices

    # Adjust extrinsic to center around the central point
    #center_point = np.average(extrinsic_matrices[:,0:3,3],axis=0)
    print(center_point.shape)
    print("center point ",center_point)
    extrinsic_matrices[:,0:3,3] -= center_point

    # Z offset assuming cameras are never below the object
    extrinsic_matrices[:,2,3] -= min(extrinsic_matrices[:,2,3].min(),0)

    # Normalize extrinsic transformation to remain within bounding box
    translation_magnitudes = np.linalg.norm(extrinsic_matrices[:,0:3,3],axis=1)
    avg_translation_magnitude = np.average(translation_magnitudes)
    print("Translation mag: ",avg_translation_magnitude)
    extrinsic_matrices[:,0:3,3] /= avg_translation_magnitude

    # scale back up TODO: make dynamic
    extrinsic_matrices[:,0:3,3] *= 4

    print("Max ",extrinsic_matrices[:,0:3,3].max())
    print("Min ",extrinsic_matrices[:,0:3,3].min())
    print("avg ",np.average(extrinsic_matrices[:,0:3,3]))

    # Convert to json
    frames = []
    for extrin, file_path in zip(extrinsic_matrices,filepaths):
        extrinsic_list = extrin.tolist()        # convert to list for json

        img_frame = { "file_path": file_path,
                          "extrinsic_matrix": extrinsic_list}

        frames.append(img_frame)

    return frames
# add the video name thing
def get_intrinsic(fp: str = "cameras.txt"):
    infile = open(fp, "r")
    lines = infile.readlines()

    read_params = False # Flag to indicate if we've successfully read params from a line
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        data = line.split(" ")
        # Filter out empty strings that can result from multiple spaces
        data = [d for d in data if d]

        if len(data) >= 8: # Need at least CAM_ID, MODEL, W, H, fx, fy, cx, cy
            try:
                camera = data[1] # MODEL
                width = int(data[2])
                height = int(data[3])
                fx = float(data[4])
                fy = float(data[5])
                x0 = float(data[6])
                y0 = float(data[7])
                read_params = True # Successfully read and parsed parameters
            except ValueError:
                # Malformed line (e.g., non-numeric value for size/params), skip
                continue
            except IndexError:
                 # Should be caught by len(data) >= 8, but as a safeguard
                continue
        
    if not read_params:
        raise ValueError("No valid camera intrinsic parameters found in file.")

    intrinsic = np.array([[fx, 0, x0], 
                          [0, fy, y0],
                          [0, 0, 1]])
    intrinsic_list = intrinsic.tolist()      # convert to list for json

    intrinsic = { "vid_width": width,
                  "vid_height": height,
                  "intrinsic_matrix": intrinsic_list
                }

    return intrinsic

# COLMAP TO NDC
def get_extrinsics_center(fp: str = "points3D.txt"):
    infile = open(fp, "r")
    lines = infile.readlines()
    point_count = 0

    central_point = np.zeros(3)
    # find center of all the points
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        data = line.split(" ")
        # Filter out empty strings
        data = [d for d in data if d]

        if len(data) >= 4: # Need at least POINT_ID, X, Y, Z
            try:
                # X Y Z
                central_point[0] += float(data[1])
                central_point[1] += float(data[2])
                central_point[2] += float(data[3])
                point_count += 1
            except ValueError:
                # Malformed line (e.g., non-numeric coordinate), skip
                continue
            except IndexError:
                # Should be caught by len(data) >=4, but as safeguard
                continue

    if point_count == 0:
        # Return NaN or raise error, current behavior is NaN due to division by zero.
        # Let's make it explicit to avoid RuntimeWarning for 0/0.
        return np.array([np.nan, np.nan, np.nan])
    else:
        central_point /= point_count
    print("Central point: ", central_point)
    return central_point


def get_json_matrices(camera_file, motion_data ):
    point_path = os.path.join(os.path.dirname(camera_file),"points3D.txt")
    center_point = get_extrinsics_center(point_path)
    intrinsic = get_intrinsic(camera_file)
    extrinsic = get_extrinsic(center_point,motion_data)
    intrinsic["frames"] = extrinsic

    return intrinsic

def main():
    # check for input argument
    if len(sys.argv) != 3:
        print("bad arguments: ")
        # python3 matrix.py images.txt camera.txt
        print("Usage: python3 %s images.txt camera.txt" % sys.argv[0])
        sys.exit(1)

    center_point = get_extrinsics_center()
    intrinsic = get_intrinsic()
    extrinsic = get_extrinsic(center_point)
    intrinsic["frames"] = extrinsic
    json_object= json.dumps(intrinsic, indent=4)

    with open('data.json', 'w') as outfile:
        outfile.write(json_object)

if __name__ == "__main__":
    image_position_extractor.extract_position_data("images.txt", "parsed_data.csv")
    main()

