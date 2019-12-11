import os
import sys
import math
import random
import argparse

import cv2
import ipdb
import scipy
import scipy.spatial
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, feature, color, transform

from vanishing_point_detection import get_vp_inliers

st = ipdb.set_trace

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

def read_images(path1, path2):
    img1 = io.imread(path1)
    img2 = io.imread(path2)
    return img1, img2

def mark_parallel_lines(img):
    plt.imshow(img)
    print("Please click")
    x = plt.ginput(12)
    plt.show()
    print('Clicks registered')
    return x

def get_feature_matches_sift(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    key_pts1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints1 = []
    for kp in key_pts1:
        keypoints1.append(np.asarray(kp.pt))
    keypoints1 = np.asarray(keypoints1)

    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    key_pts2, descriptors2 = sift.detectAndCompute(gray2,None)
    keypoints2 = []
    for kp in key_pts2:
        keypoints2.append(np.asarray(kp.pt))
    keypoints2 = np.asarray(keypoints2)

    matches = []
    matches_cv2 = matcher.knnMatch(descriptors1, descriptors2, 2)
    for i, (m1, m2) in enumerate(matches_cv2):
        if m1.distance < 0.7 * m2.distance:
            matches.append(np.asarray([m1.queryIdx, m1.trainIdx]))

    matches = np.asarray(matches)
    return [keypoints1, keypoints2], [descriptors1, descriptors2], matches

def visualize_matches(img1, img2, keypoints, matches):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.gray()
    feature.plot_matches(ax, img1, img2, keypoints[0], keypoints[1], matches, only_matches=True, alignment='vertical')
    ax.axis('off')
    ax.set_title("Original Image vs. Transformed Image")
    plt.show()
    plt.close()

def get_vanishing_points(parallel_pts):
    line1 = np.cross(np.append(np.asarray(parallel_pts[0]), 1), np.append(np.asarray(parallel_pts[1]), 1))
    line2 = np.cross(np.append(np.asarray(parallel_pts[2]), 1), np.append(np.asarray(parallel_pts[3]), 1))
    vp1 = np.cross(line1, line2)
    vp1 = vp1 / vp1[-1]

    line3 = np.cross(np.append(np.asarray(parallel_pts[4]), 1), np.append(np.asarray(parallel_pts[5]), 1))
    line4 = np.cross(np.append(np.asarray(parallel_pts[6]), 1), np.append(np.asarray(parallel_pts[7]), 1))
    vp2 = np.cross(line3, line4)
    vp2 = vp2 / vp2[-1]

    line5 = np.cross(np.append(np.asarray(parallel_pts[8]), 1), np.append(np.asarray(parallel_pts[9]), 1))
    line6 = np.cross(np.append(np.asarray(parallel_pts[10]), 1), np.append(np.asarray(parallel_pts[11]), 1))
    vp3 = np.cross(line5, line6)
    vp3 = vp3 / vp3[-1]

    return vp1, vp2, vp3

def compute_w(vp1, vp2, vp3):
    a0 = np.array([[ (vp1[0]*vp2[0]) + (vp1[1] * vp2[1]) ],
                  [ vp1[1] + vp1[0] ],
                  [ vp2[1] + vp1[1] ],
                  [1]])
    a1 = np.array([[ (vp1[0]*vp3[0]) + (vp1[1] * vp3[1]) ],
                  [ vp1[1] + vp1[0] ],
                  [ vp3[1] + vp1[1] ],
                  [1]])

    a2 = np.array([[ (vp3[0]*vp2[0]) + (vp3[1] * vp2[1]) ],
                  [ vp3[1] + vp3[0] ],
                  [ vp2[1] + vp3[1] ],
                  [1]])

    A = np.array([a0, a1, a2]).squeeze()
    U, S, V = np.linalg.svd(A)
    soln = V[-1]
    W = np.array([[soln[0], 0, soln[1]],[0, soln[0], soln[2]],[soln[1], soln[2], soln[3]]])
    return W

def decompose_to_K(W):
    K = np.linalg.inv(np.linalg.cholesky(W).T)
    return K

def refineF(F, pts1, pts2):
    def _singularize(F):
        U, S, V = np.linalg.svd(F)
        S[-1] = 0
        F = U.dot(np.diag(S).dot(V))
        return F

    def _objective_F(f, pts1, pts2):
        F = _singularize(f.reshape([3, 3]))
        num_points = pts1.shape[0]
        hpts1 = np.concatenate([pts1, np.ones([num_points, 1])], axis=1)
        hpts2 = np.concatenate([pts2, np.ones([num_points, 1])], axis=1)
        Fp1 = F.dot(hpts1.T)
        FTp2 = F.T.dot(hpts2.T)

        r = 0
        for fp1, fp2, hp2 in zip(Fp1.T, FTp2.T, hpts2):
            r += (hp2.dot(fp1))**2 * (1/(fp1[0]**2 + fp1[1]**2) + 1/(fp2[0]**2 + fp2[1]**2))
        return r
    f = scipy.optimize.fmin_powell(
        lambda x: _objective_F(x, pts1, pts2), F.reshape([-1]),
        maxiter=100000,
        maxfun=10000,
        disp=False
    )
    return _singularize(f.reshape([3, 3]))

def compute_essential_matrix(F, K1, K2):
    return K2.T.dot(F).dot(K1)

def triangulate(C1, pts1, C2, pts2):
    projected_pts = []
    for i in range(len(pts1)):
        x1 = pts1[i,0]
        y1 = pts1[i,1]
        x2 = pts2[i,0]
        y2 = pts2[i,1]
        A = np.array([x1 * C1[2].T - C1[0].T, y1 * C1[2].T - C1[1].T, x2 * C2[2].T - C2[0].T, y2 * C2[2].T - C2[1].T])
        U, S, V = np.linalg.svd(A)
        ppts = V[-1]
        ppts = ppts / ppts[-1]
        projected_pts.append(ppts)
    projected_pts = np.asarray(projected_pts)
    projected_pts1 = C1.dot(projected_pts.T)
    projected_pts2 = C2.dot(projected_pts.T)
    projected_pts1 = projected_pts1 / projected_pts1[-1,:]
    projected_pts2 = projected_pts2 / projected_pts2[-1,:]
    projected_pts1 = projected_pts1[:2,:].T
    projected_pts2 = projected_pts2[:2,:].T
    error = np.linalg.norm(projected_pts1 - pts1, axis=-1)**2 + np.linalg.norm(projected_pts2 - pts2, axis=-1)**2
    error = error.sum()
    return projected_pts[:,:3], error

def visualize_3d(pts_3d):

    # x, y, z = pts_3d[:,0], pts_3d[:,1], pts_3d[:,2]
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # plt.gca().set_aspect('equal', adjustable='box')
    # ax.scatter(x, y, z, color='blue')
    # plt.show()
    # plt.close()

    total_points_list = []
    total_points_list.append(go.Scatter3d(
            x=pts_3d[:,0].tolist(),
            y=pts_3d[:,2].tolist(),
            z=(-pts_3d[:,1]).tolist(),
            mode='markers',
            marker=dict(
                size=5,
                color='firebrick',
                line=dict(
                    color='black',
                    width=0.1
                ),
                opacity=0.8
            )
        )
    )

    plotly.offline.plot({
            "data": total_points_list,
            "layout": go.Layout(title="All Planes")
        }, auto_open=True)

def visualize_3d_planes(pts_3d_plane_1, pts_3d_plane_2):

    total_points_list = []
    total_points_list.append(go.Scatter3d(
            x=pts_3d_plane_1[:,0].tolist(),
            y=pts_3d_plane_1[:,2].tolist(),
            z=(-pts_3d_plane_1[:,1]).tolist(),
            mode='markers',
            marker=dict(
                size=5,
                color='lightcoral',
                line=dict(
                    color='black',
                    width=0.1
                ),
                opacity=1.0
            )
        )
    )

    total_points_list.append(go.Scatter3d(
            x=pts_3d_plane_2[:,0].tolist(),
            y=pts_3d_plane_2[:,2].tolist(),
            z=(-pts_3d_plane_2[:,1]).tolist(),
            mode='markers',
            marker=dict(
                size=5,
                color='limegreen',
                line=dict(
                    color='black',
                    width=0.1
                ),
                opacity=1.0
            )
        )
    )

    plotly.offline.plot({
            "data": total_points_list,
            "layout": go.Layout(title="Plane Fitting")
        }, auto_open=True)


def visualize_3d_color(pts_3d, img, key_pts, pts_old=None):
    key_pts = np.rint(key_pts).astype('int')
    kp_x = np.clip(key_pts[:,0], 0, img.shape[0]-1)
    kp_y = np.clip(key_pts[:,1], 0, img.shape[1]-1)
    r = img[:,:,0][kp_y, kp_x]
    g = img[:,:,1][kp_y, kp_x]
    b = img[:,:,2][kp_y, kp_x]

    colors = np.array([r,g,b]).T / 255.0

    # fig = plt.figure(figsize=(8,8))
    # ax = fig.add_subplot(111, projection='3d')
    # for p, c in zip(pts_3d, colors):
    #     ax.plot([p[0]], [p[1]], [p[2]], '.', color=(c[0], c[1], c[2]), markersize=8, alpha=0.5)  

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()
    # plt.close()

    total_points_list = []
    total_points_list.append(go.Scatter3d(
            x=pts_3d[:,0].tolist(),
            y=pts_3d[:,1].tolist(),
            z=(pts_3d[:,2]).tolist(),
            mode='markers',
            marker=dict(
                size=5,
                line=dict(
                    color=colors,
                    width=0.1
                ),
                opacity=1
            )
        )
    )

    # total_points_list.append(go.Scatter3d(
    #             x=pts_old[0].tolist(),
    #             y=pts_old[1].tolist(),
    #             z=(pts_old[2]).tolist(),
    #             mode='markers',
    #             marker=dict(
    #                 size=2,
    #                 line=dict(
    #                     color='rgba(100, 100, 100, 0.14)',
    #                     width=0.1
    #                 ),
    #                 opacity=1
    #             )
    #         )
    #     )

    plotly.offline.plot({
                "data": total_points_list,
                "layout": go.Layout(title="3D Reconstruction")
            }, auto_open=True)


def get_plane(x, y, z):
    A = np.stack((x,y,z,np.ones(x.shape[0])), 1)
    u,s,v = np.linalg.svd(A)
    return v[-1]

def run_plane_ransac(x, y, z, ignore_pts=None):
    tol = 1e-2
    n_iter = 10000
    max_inliers = 0
    data = np.stack((x,y,z,np.ones(x.shape[0])), 1)
    if ignore_pts is None:
        ignore_pts = np.zeros((x.shape[0])).astype('bool')
        idx_to_choose = np.arange(x.shape[0])
    else:
        idx_to_choose = np.where(ignore_pts==0)[0]
    for i in range(n_iter):
        chosen_idx = np.random.choice(idx_to_choose, 4, replace=False)
        x_rand, y_rand, z_rand = x[chosen_idx], y[chosen_idx], z[chosen_idx]
        predicted_plane = get_plane(x_rand, y_rand, z_rand)
        predicted_plane_distance = (data * predicted_plane).sum(axis=1) / np.sqrt(np.square(predicted_plane[:-1]).sum())
        inliers = np.abs(predicted_plane_distance) < tol
        inliers[ignore_pts] = False
        inliers_count = inliers.sum()
        if inliers_count > max_inliers:
            max_inliers = inliers_count
            best_inliers = inliers
    p = np.where(best_inliers==True)
    x_best, y_best, z_best = x[p], y[p], z[p]
    best_plane = get_plane(x_best, y_best, z_best)
    return best_plane, best_inliers

def visualize_plane_fitting(plane_list, x, y, z):
    X_final = []
    Y_final = []
    Z_final = []
    k1, k2, k3, k4 = -1.5,1.5,-0.5,1.5
    for plane in plane_list:
        a, b, c, d = plane
        p = np.linspace(k1,k2,20)
        q = np.linspace(k3,k4,20)
        X,Y = np.meshgrid(p,q)
        Z = (-d - a*X - b*Y) / c
        X_final.append(X)
        Y_final.append(Y)
        Z_final.append(Z)

    fig = plt.figure()
    ax = Axes3D(fig)
    for idx in range(len(plane_list)):
        ax.plot_surface(X_final[idx], Y_final[idx], Z_final[idx], alpha=0.2)
    ax.scatter(x, y, z, color='green')
    plt.show()
    plt.close()

def ransacF(pts1, pts2, M):
    def _sevenpoint(pts1, pts2, M):
        pts1 = pts1.astype('float64')
        pts2 = pts2.astype('float64')
        pts1 /= M
        pts2 /= M
        A = [pts2[:,0]*pts1[:,0], pts2[:,0]*pts1[:,1], pts2[:,0], pts2[:,1]*pts1[:,0], pts2[:,1]*pts1[:,1], pts2[:,1], pts1[:,0], pts1[:,1], np.ones_like(pts1[:,0])]
        A = np.asarray(A).T
        U, S, V = np.linalg.svd(A)
        F1 = V[-1].reshape(3,3)
        F2 = V[-2].reshape(3,3)
        func = lambda x: np.linalg.det((x * F1) + ((1 - x) * F2))
        x0 = func(0)
        x1 = (2 * (func(1) - func(-1)) / 3.0) - ((func(2) - func(-2)) / 12.0)
        x2 = (0.5 * func(1)) + (0.5 * func(-1)) - func(0)
        x3 = func(1) - x0 - x1 - x2
        alphas = np.roots([x3,x2,x1,x0])
        alphas = np.real(alphas[np.isreal(alphas)])
        final_F = []
        for a in alphas:
            F = (a * F1) + ((1 - a) * F2)
            F = refineF(F, pts1, pts2)
            t = np.array([[1.0/M,0.0,0.0],[0.0,1.0/M,0.0],[0.0,0.0,1.0]])
            F = t.T.dot(F).dot(t)
            final_F.append(F)
        return final_F

    def _eightpoint(pts1, pts2, M):
        pts1 = pts1.astype('float64')
        pts2 = pts2.astype('float64')
        pts1 /= M
        pts2 /= M
        A = [pts2[:,0]*pts1[:,0], pts2[:,0]*pts1[:,1], pts2[:,0], pts2[:,1]*pts1[:,0], pts2[:,1]*pts1[:,1], pts2[:,1], pts1[:,0], pts1[:,1], np.ones_like(pts1[:,0])]
        A = np.asarray(A).T
        U, S, V = np.linalg.svd(A)
        F = V[-1].reshape(3,3)
        U, S, V = np.linalg.svd(F)
        S[-1] = 0.0
        F = U.dot(np.diag(S)).dot(V)
        F = refineF(F, pts1, pts2)
        t = np.array([[1.0/M,0.0,0.0],[0.0,1.0/M,0.0],[0.0,0.0,1.0]])
        F = t.T.dot(F).dot(t)
        return F

    num_iter = 200
    threshold = 1e-3
    total = len(pts1)
    max_inliers = 0
    for i in range(num_iter):
        idx = np.random.permutation(np.arange(total))[:7]
        selected_pts1, selected_pts2 = pts1[idx], pts2[idx]
        F7 = _sevenpoint(selected_pts1, selected_pts2, M)
        for k, F in enumerate(F7):
            pts1_homo = np.concatenate((pts1, np.ones((pts1.shape[0],1))), axis=-1)
            pts2_homo = np.concatenate((pts2, np.ones((pts2.shape[0],1))), axis=-1)
            error = []
            for p, q in zip(pts1_homo, pts2_homo):
                error.append(q.T.dot(F).dot(p))
            error = np.abs(np.asarray(error))
            inliers = error < threshold

            if inliers.sum() > max_inliers:
                max_inliers = inliers.sum()
                best_inliers = inliers
                best_k = k
    selected_pts1, selected_pts2 = pts1[best_inliers], pts2[best_inliers]
    F = _eightpoint(selected_pts1, selected_pts2, M)
    return F

def get_3d_pts(E, pts1, pts2, K1, K2):
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K1)

    M1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    M2 = np.hstack((R, t))

    C1 = np.dot(K1,  M1)
    C2 = np.dot(K2,  M2)
    P, _ = triangulate(C1, pts1, C2, pts2)
    val = 6
    flag1 = (P < -val)
    flag2 = (P > val)
    outliers = np.logical_or(flag1, flag2).any(axis=-1)
    P = P[~outliers]
    # outliers_ = P[:,2]<12
    # P = P[~outliers_]
    return P, C1, C2, [pts1, pts2]

def autocalibrate(load_or_label, image_path_1):
    if load_or_label:
        line_path_1 = 'parallel_img1_{}'.format(os.path.basename(image_path_1).split('.')[0]) + '.npy'
        if os.path.exists(line_path_1):
            parallel_pts_1 = np.load(line_path_1)
            vp1, vp2, vp3 = get_vanishing_points(parallel_pts_1)
        else:
            parallel_pts_1 = mark_parallel_lines(image_1)
            np.save(line_path_1, parallel_pts_1)
            vp1, vp2, vp3 = get_vanishing_points(parallel_pts_1)
    else:
        vp_path_1 = '{}_vp'.format(os.path.basename(image_path_1).split('.')[0]) + '.npy'
        if os.path.exists(vp_path_1):
            hypothesis_list = np.load(vp_path_1)
            vp1, vp2, vp3 = hypothesis_list
        else:
            _, hypothesis_list, _ = get_vp_inliers(image_path_1, 5, 3000, 11, 7, 2)
            # vp1, vp2, vp3 = getVP(image_path_1)
            # hypothesis_list = np.array([vp1, vp2, vp3])

            hypothesis_list = np.asarray(hypothesis_list)
            np.save(vp_path_1, hypothesis_list)
            vp1, vp2, vp3 = hypothesis_list

    W1 = compute_w(vp1, vp2, vp3)
    K1 = decompose_to_K(W1)
    K2 = K1
    return K1, K2

def get_convex_hull_pts(projected_points_1, projected_points_2, inliers, image_1, image_2):
    first_plane_inlier_pts_1 = projected_points_1.T[inliers]
    first_plane_inlier_pts_1 = first_plane_inlier_pts_1 / first_plane_inlier_pts_1[:,-1].reshape(first_plane_inlier_pts_1.shape[0], 1)
    pts_1 = first_plane_inlier_pts_1[:,:2]
    hull_1 = scipy.spatial.ConvexHull(pts_1)
    hull_path_1 = Path(pts_1[hull_1.vertices])

    first_plane_inlier_pts_2 = projected_points_2.T[inliers]
    first_plane_inlier_pts_2 = first_plane_inlier_pts_2 / first_plane_inlier_pts_2[:,-1].reshape(first_plane_inlier_pts_2.shape[0], 1)
    pts_2 = first_plane_inlier_pts_2[:,:2]
    hull_2 = scipy.spatial.ConvexHull(pts_2)
    hull_path_2 = Path(pts_2[hull_2.vertices])
    # st()
    plt.imshow(image_1)
    plt.plot(pts_1[hull_1.vertices,0], pts_1[hull_1.vertices,1], 'r--', lw=2)
    plt.plot(pts_1[hull_1.vertices[0],0], pts_1[hull_1.vertices[0],1], 'ro')
    plt.show()
    plt.close()

    plane_1_img_1_pts = []
    for i in range(image_1.shape[0]):
        for j in range(image_1.shape[1]):
            img_pt = np.array([i, j])
            if hull_path_1.contains_point(img_pt):
                plane_1_img_1_pts.append(img_pt)

    plane_1_img_2_pts = []
    for i in range(image_2.shape[0]):
        for j in range(image_2.shape[1]):
            img_pt = np.array([i, j])
            if hull_path_2.contains_point(img_pt):
                plane_1_img_2_pts.append(img_pt)

    # plane_1_img_1_pts = np.asarray(plane_1_img_1_pts)
    # fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
    # ax = axes.ravel()
    # cmap = plt.get_cmap('tab20')
    # my_colors = ['ro', 'go', 'bo', 'ko', 'yo', 'mo', 'co'] * 1000
    # ax[0].imshow(image_1)
    # t = 0
    # for inn in range(plane_1_img_1_pts.shape[0]):
    #     if inn % 100 == 0:
    #         t += 1
    #         ax[0].plot([plane_1_img_1_pts[inn][1]],[plane_1_img_1_pts[inn][0]], my_colors[t])

    # for a in ax:
    #     a.set_axis_off()
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    return np.asarray(plane_1_img_1_pts), np.asarray(plane_1_img_2_pts)

def compute_homography(projected_points_1, projected_points_2, dominant_plane_inliers, num_iter=5000, tol=0.5):
    def _compute_h(p1, p2):
        assert(p1.shape[1]==p2.shape[1])
        assert(p1.shape[0]==2)
        N = p1.shape[1]
        A = np.zeros((2*N,9))

        x, y = p1[0], p1[1]
        u, v = p2[0], p2[1]

        A[::2,0] = 0
        A[::2,1] = 0
        A[::2,2] = 0

        A[1::2,0] = u
        A[1::2,1] = v
        A[1::2,2] = 1

        A[::2,3] = -u
        A[::2,4] = -v
        A[::2,5] = -1

        A[1::2,3] = 0
        A[1::2,4] = 0
        A[1::2,5] = 0

        A[::2,6] = u * y
        A[::2,7] = v * y
        A[::2,8] = y

        A[1::2,6] = -(x * u)
        A[1::2,7] = -(x * v)
        A[1::2,8] = -x

        U, S, V = np.linalg.svd(A)
        H2to1 = V[-1].reshape(3,3)
        return H2to1

    projected_points_1 = projected_points_1.T
    projected_points_2 = projected_points_2.T
    pt1 = projected_points_1[dominant_plane_inliers]
    pt2 = projected_points_2[dominant_plane_inliers]
    pt1_homo = pt1 / pt1[:,-1].reshape(pt1.shape[0], 1)
    pt2_homo = pt2 / pt2[:,-1].reshape(pt2.shape[0], 1)
    max_inliers = 0
    for i in range(num_iter):
        rand_idx = np.random.choice(pt1_homo.shape[0], 4, replace=False)
        p1 = pt1_homo[rand_idx]
        p2 = pt2_homo[rand_idx]
        p1 = p1[:,:2].T
        p2 = p2[:,:2].T
        H = _compute_h(p1, p2)
        p1_transformed = np.matmul(H, pt2_homo.T)
        p1_transformed = p1_transformed / p1_transformed[-1,:]
        error = np.linalg.norm(p1_transformed.T - pt1_homo, axis=-1) < tol
        inliers = error.sum()
        if inliers > max_inliers:
            max_inliers = inliers
            best_error = error

    p1 = pt1_homo[np.where(best_error==True)][:,:2].T
    p2 = pt2_homo[np.where(best_error==True)][:,:2].T
    bestH = _compute_h(p1, p2)
    return bestH

def get_planar_correspondences(plane_pts, H):
    plane_pts_homo = np.concatenate((plane_pts, np.ones((len(plane_pts),1))), axis=-1)
    correspondences = H.dot(plane_pts_homo.T).T
    correspondences = correspondences / correspondences[:,-1].reshape(correspondences.shape[0], 1)
    correspondences = correspondences[:,:2]
    return correspondences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f1', '--image_path_1', default=None, type=str, help='Full image path for left image')
    parser.add_argument('-f2', '--image_path_2', default=None, type=str, help='Full image path for right image')
    parser.add_argument('-l', '--load_or_label', default=False, type=bool, help='True if manual label for vp')
    args = parser.parse_args()
    if args.image_path_1 is None or args.image_path_2 is None:
        raise ValueError('Run python detect_vanishing_point.py -h for command line arguments')
    image_path_1 = args.image_path_1
    image_path_2 = args.image_path_2
    load_or_label = args.load_or_label

    print('Getting SIFT features')
    image_1, image_2 = read_images(image_path_1, image_path_2)
    keypoints, descriptors, matches = get_feature_matches_sift(image_1, image_2)
    pts1, pts2 = keypoints[0][matches[:,0]], keypoints[1][matches[:,1]]
    # visualize_matches(image_1, image_2, keypoints, matches)

    print('Autocalibrating')
    K1, K2 = autocalibrate(load_or_label, image_path_1)

    print('Computing E, F and P')
    scale_normalizer = max(image_2.shape)
    F = ransacF(pts1, pts2, scale_normalizer)
    E = compute_essential_matrix(F, K1, K2)
    P, C1, C2, pts_temp = get_3d_pts(E, pts1, pts2, K1, K2)
    visualize_3d(P)

    print('Fitting planes in 3D')
    x, y, z = P[:,0], P[:,1], P[:,2]
    fitted_plane_1, inliers_1 = run_plane_ransac(x, y, z)
    ignore_pts = inliers_1
    fitted_plane_2, inliers_2 = run_plane_ransac(x, y, z, ignore_pts=ignore_pts)
    ignore_pts = np.logical_or(inliers_1, inliers_2)
    fitted_plane_3, inliers_3 = run_plane_ransac(x, y, z, ignore_pts=ignore_pts)
    dominant_plane_inliers = [inliers_1, inliers_2]
    # visualize_plane_fitting([fitted_plane_1, fitted_plane_2], x, y, z)
    P_plane1, P_plane2 = P[inliers_1], P[inliers_2]
    visualize_3d_planes(P_plane1, P_plane2)

    print('Projecting plane inliers to image')
    ones = np.ones((x.shape[0])).reshape(1, x.shape[0])
    points_3d = np.concatenate((np.asarray([x, y, z]), ones), axis=0)
    projected_points_1 = C1.dot(points_3d)
    projected_points_2 = C2.dot(points_3d)

    print('Computing planar homographies')
    plane_1_homography_1to2 = compute_homography(projected_points_2, projected_points_1, dominant_plane_inliers[0])
    plane_2_homography_1to2 = compute_homography(projected_points_2, projected_points_1, dominant_plane_inliers[1])

    print('Generating convex hull')
    plane_1_img_1_pts, plane_1_img_2_pts = get_convex_hull_pts(projected_points_1, projected_points_2,
                                                       dominant_plane_inliers[0], image_1, image_2)

    plane_2_img_1_pts, plane_2_img_2_pts = get_convex_hull_pts(projected_points_1, projected_points_2,
                                                       dominant_plane_inliers[1], image_1, image_2)

    print('Getting correspondences')
    plane_1_img_2_pts_corr = get_planar_correspondences(plane_1_img_1_pts, plane_1_homography_1to2)
    plane_2_img_2_pts_corr = get_planar_correspondences(plane_2_img_1_pts, plane_2_homography_1to2)

    print('Reconstructing dense points in 3D')
    plane_1_3d_pts, _ = triangulate(C1, plane_1_img_1_pts.astype('float64'), C2, plane_1_img_2_pts_corr)
    plane_2_3d_pts, _ = triangulate(C1, plane_2_img_1_pts.astype('float64'), C2, plane_2_img_2_pts_corr)
    points_3d_all = np.concatenate((plane_1_3d_pts, plane_2_3d_pts), axis=0)
    points_2d_all = np.concatenate((plane_1_img_1_pts, plane_2_img_1_pts), axis=0)
    visualize_3d_color(points_3d_all, image_1, points_2d_all, [x, y, z])


if __name__ == '__main__':
    main()
    