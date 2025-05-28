from .to_cam import CameraPoseVisualizer
import json
import sys
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#Performs the RANSAC algorithm on a set of points with threshold tau for inliers
"""
def ransac(lines,samples,tau=0.5):
	N=len(lines)
	best_fitting_inliers = 0
	best_outlier_dist =0
	best_inlier_dist =0
	best_inlier_mask=None
	FOE=None

	# Outer RANSAC loop
	for i in range(samples):
		# Randomly selects each sample point skipping if it is the same point
		sample = np.random.randint(0,N,2)
		if sample[0] == sample[1]:
			continue

		#calculate intersecting point
		print("l1",lines[sample[0]])
		print("l2",lines[sample[1]])
		P = np.cross(lines[sample[0]], lines[sample[1]])
		x_pnt = P[0]/(P[2]+0.00000001)
		y_pnt = P[1]/(P[2]+0.00000001)

		#distances of each point to the line exploiting the fact that the n_vector is normal to the line
		distances = abs((x_pnt*lines[:,0]) + (y_pnt*lines[:,1]) +lines[:,2])

		#recording inliers and outliers based on the distance and threshhold tau
		inliers = np.sum(np.where(distances<tau,1,0))
		outliers = N-inliers
		inlier_dist = np.sum(np.where(distances<tau,distances,0))
		outlier_dist = np.sum(np.where(distances<tau,0,distances))
		#intlier_mask = np.where(distances<tau,1,0)

		#recording and printing stats on the current best fit line based on inlier ammount
		if inliers>best_fitting_inliers:
			best_fitting_inliers = inliers
			best_outlier_dist = outlier_dist
			best_inlier_dist = inlier_dist

			best_inlier_mask = distances<tau
			FOE = (x_pnt,y_pnt)

	#final statistics for inliers and outliers
	avg_inlier_dist = best_inlier_dist/(best_fitting_inliers)
	return FOE, best_inlier_mask, avg_inlier_dist
    """
def closest_point_to_rays(a,d):
    """
    Find the nearest point to a set of rays.

    Args:
    p: a numpy array of shape (3,) representing the nearest point to the rays
    a: a numpy array of shape (n, 3) representing the origins of the rays
    d: a numpy array of shape (n, 3) representing the directions of the rays
    n: an integer representing the number of rays

    Returns:
    None. The function calculates the nearest point to the rays and stores it in p.
    """
    n = len(a)
    m = np.zeros((3, 3))
    b = np.zeros(3)

    for i in range(n):
        d2 = np.dot(d[i], d[i])
        da = np.dot(d[i], a[i])

        for ii in range(3):
            for jj in range(3):
                m[ii][jj] += d[i][ii] * d[i][jj]

            m[ii][ii] -= d2
            b[ii] += d[i][ii] * da - a[i][ii] * d2

    x =np.linalg.solve(m, b)
    print(x)
    return x


if __name__ =='__main__':
    print("Starting Recentering")
    input_file = sys.argv[1]

    input_str = open(input_file)
    input = json.loads(input_str.read())

    extrins = []
    intrins = np.array(input["intrinsic_matrix"])
    for f in input["frames"]:
        extrinsic = np.array(f["extrinsic_matrix"])
        #c_pos = extrinsic[0:3,3]

        extrins+=[ extrinsic ]

    visualizer = CameraPoseVisualizer([-5, 5], [-5, 5], [0, 5])
    cams = []
    ray_o = [] # ray origins
    ray_d = [] # ray directions
    avg = np.zeros(3)
    for i,e in enumerate(extrins):
        if("object_center" in input["frames"][i]):
        #if i%3 == 0:
            color = plt.cm.rainbow(i / len(extrins))
            visualizer.extrinsic2pyramid(e, color)
            primary_point = np.asarray([0,0,-2,1])

            r = e[0:3,0:3]
            t = e[0:3,3]
            avg+=t
            c = -r.T @ t
            print("Rotation:\n",r)
            print("Translation:\n",t)
            print("Cam:\n",c)
            print()
            #visualizer.plot_cam(e @ primary_point, color)
            #secondary_point = np.asarray([0,0,-3,1])
            #visualizer.plot_cam(e @ secondary_point, color)

            #twodinput = np.array([[400],[400],[1]])
            print("OBJECT CENTER: ",input["frames"][i]["object_center"])
            #twodinput = 2*intrins[:,2] [:, np.newaxis]
            #twodinput = np.append(np.asarray(input["frames"][i]["object_center"]), [1])[:, np.newaxis]
            #oneVector = np.array([[0],[0],[1]])
            #product = e @ np.append(np.linalg.inv(intrins) @ twodinput, [1])[ np.newaxis, :].T
            #product=product.T[0]
            #product=product[0:3]
            #visualizer.plot_ray(t, -5*product+t)

            twodinput = np.append(np.asarray(input["frames"][i]["object_center"]), [1])[:, np.newaxis]
            oneVector = np.array([[0],[0],[1]])
            product = r @ np.linalg.inv(intrins) @ twodinput
            product=product.T[0]
            product = product / np.linalg.norm(product)
            visualizer.plot_ray(t, -5*product+t,"green")

            
            ray_o.append(t)
            ray_d.append(product)

            fov = True
            if fov:
                twodinput = intrins[:,2] [:, np.newaxis]
                product = r @ np.linalg.inv(intrins) @ twodinput
                product=product.T[0]
                product = product / np.linalg.norm(product)
                visualizer.plot_ray(t, -1*product+t,"blue")

                #twodinput = np.array([[0],[0],[1]])
                twodinput = np.array([[-intrins[0,2]],[-intrins[1,2]],[1]])
                product = r @ np.linalg.inv(intrins) @ twodinput
                product=product.T[0]
                product = product / np.linalg.norm(product)
                visualizer.plot_ray(t, -1*product+t,"blue")

                twodinput = np.array([[intrins[0,2]],[0],[1]])
                product = r @ np.linalg.inv(intrins) @ twodinput
                product=product.T[0]
                product = product / np.linalg.norm(product)
                visualizer.plot_ray(t, -1*product+t,"blue")

                twodinput = np.array([[0],[intrins[1,2]],[1]])
                product = r @ np.linalg.inv(intrins) @ twodinput
                product=product.T[0]
                product = product / np.linalg.norm(product)
                visualizer.plot_ray(t, -1*product+t,"blue")

        #offset = np.asarray([0,0.75,-0.75])
        #scale = .8
        # x
        #e[0:3,3] = (scale*e[0:3,3] + offset)
        #input["frames"][i]["extrinsic_matrix"] = e.tolist()
    print("Finding Object Center")
    ray_d = np.asarray(ray_d)
    ray_d = ray_d/np.linalg.norm(ray_d,axis=1)[:,np.newaxis]
    #n_vec = n_vec/np.linalg.norm(n_vec,axis=1).reshape(-1,1)
    #n_vec = n_vec[:,::-1]
    #n_vec[:,0] = -n_vec[:,0]

    ray_o = np.asarray(ray_o)

    object_center = closest_point_to_rays(ray_d,ray_o)*10
    print("Object_Center",object_center)
    
    #scale = 0.4
    scale = 0.6
    for i in range(len(input["frames"])):
        e = extrins[i]
        #offset = np.asarray([0,0.5,-0.25])
        #offset = np.asarray([-1.5,-1.25,0.25])\
        offset = np.asarray([-0.75,2.5,2.0])
        e[0:3,3] = scale*(e[0:3,3]  + offset)#+object_center) #object_center
        print(e[0:3,3])
        input["frames"][i]["extrinsic_matrix"] = e.tolist()

    with open("new_"+input_file, "w") as outfile:
        json.dump(input, outfile, indent=4)
    print("Center",object_center)
    visualizer.show()

