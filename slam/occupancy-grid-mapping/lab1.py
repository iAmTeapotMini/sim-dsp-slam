#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import bresenham as bh

def plot_gridmap(gridmap):
    plt.figure()
    plt.imshow(gridmap, cmap='Greys',vmin=0, vmax=1)
    
def init_gridmap(size, res):
    gridmap = np.zeros([int(np.ceil(size/res)), int(np.ceil(size/res))])
    return gridmap

def world2map(pose, gridmap, map_res):
    origin = np.array(gridmap.shape)/2
    new_pose = np.zeros(2)
    new_pose[0] = np.round(pose[0]/map_res) + origin[0]
    new_pose[1] = np.round(pose[1]/map_res) + origin[1]
    return new_pose.astype(np.int32)

def v2t(pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return tr    

def ranges2points(ranges):
    # laser properties
    start_angle = -1.5708
    angular_res = 0.0087270
    max_range = 30
    # rays within range
    num_beams = ranges.shape[0]
    idx = (ranges < max_range) & (ranges > 0)
    # 2D points
    angles = np.linspace(start_angle, start_angle + (num_beams*angular_res), num_beams)[idx]
    points = np.array([np.multiply(ranges[idx], np.cos(angles)), np.multiply(ranges[idx], np.sin(angles))])
    # homogeneous points
    points_hom = np.append(points, np.ones((1, points.shape[1])), axis=0)
    return points_hom

def ranges2cells(r_ranges, w_pose, gridmap, map_res):
    # ranges to points
    r_points = ranges2points(r_ranges)
    w_P = v2t(w_pose)
    w_points = np.matmul(w_P, r_points)
    # covert to map frame
    m_points = np.array([world2map(w_point, gridmap, map_res) for w_point in w_points])
    m_points = m_points[0:2,:]
    return m_points

def poses2cells(w_pose, gridmap, map_res):
    # covert to map frame
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose  

def bresenham(x0, y0, x1, y1):
    l = np.array(list(bh.bresenham(x0, y0, x1, y1)))
    return l
    
def prob2logodds(p):
    # epsilon = 1e-12  # очень маленькое число
    # p = np.clip(p, epsilon, 1 - epsilon) 
    return np.log(p/(1-p))
    
def logodds2prob(l):
    return 1/(1 + np.exp(-l)) 
    
def inv_sensor_model(cell, endpoints, prob_occ, prob_free):
    # prob_occ - вероятность занятости ячейки
    # prob_free - вероятность, что ячейка свободна, когда она находится в области покрытия наблюдения Zt
    if cell in endpoints:
        return prob2logodds(prob_occ)
    return prob2logodds(prob_free)

def grid_mapping_with_known_poses(ranges_raw, poses_raw, occ_gridmap, map_res, prob_occ, prob_free, prior):
    
    l0 = prob2logodds(prior)
    
    for r, p in zip(ranges_raw, poses_raw):
        pos_cell = poses2cells(p, occ_gridmap, map_res)
        ranges_points = ranges2points(r)
        ranges_cells = ranges2cells(r, p, occ_gridmap, map_res)
        visibility_cells = [bresenham(pos_cell[0], pos_cell[1], ci[0], ci[1]) for ci in ranges_cells.T]
        visibility_cells = np.unique(np.vstack(visibility_cells), axis=0)
        for mi in visibility_cells:
            l = prob2logodds(occ_gridmap[mi[0], mi[1]]) + inv_sensor_model(mi, ranges_cells, prob_occ, prob_free) - l0
            occ_gridmap[mi[0], mi[1]] = logodds2prob(l)
    return occ_gridmap