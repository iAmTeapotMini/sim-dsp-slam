from scipy.stats import multivariate_normal

from lab3.main import angle_normalization
from read_data import read_world, read_sensor_data
from misc_tools import *
import numpy as np
import math
import copy


def normalize_angle(angle):
    """Привести угол к диапазону [-pi, pi]"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def initialize_particles(num_particles, num_landmarks):
    #initialize particle at pose [0,0,0] with an empty map

    particles = []

    for i in range(num_particles):
        particle = dict()

        #initialize pose: at the beginning, robot is certain it is at [0,0,0]
        particle['x'] = 0
        particle['y'] = 0
        particle['theta'] = 0

        #initial weight
        particle['weight'] = 1.0 / num_particles
        
        #particle history aka all visited poses
        # particle['history'] = []

        #initialize landmarks of the particle
        landmarks = dict()

        for i in range(num_landmarks):
            landmark = dict()

            #initialize the landmark mean and covariance 
            landmark['mu'] = [0,0]
            landmark['sigma'] = np.zeros([2,2])
            landmark['observed'] = False

            landmarks[i+1] = landmark

        #add landmarks to particle
        particle['landmarks'] = landmarks

        #add particle to set
        particles.append(particle)

    return particles

def sample_motion_model(odometry, particles):
    # Updates the particle positions, based on old positions, the odometry
    # measurements and the motion noise 

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    a1, a2, a3, a4 = [0.1, 0.1, 0.05, 0.05]

    for p in particles:
        # p['history'].append({'x':p['x'], 'y':p['y'], 'theta':p['theta']})

        cap_delta_rot1 = delta_rot1 + np.random.normal(size=1, loc=0, scale=np.sqrt(a1*abs(delta_rot1) + a2*delta_trans))
        cap_delta_trans = delta_trans + np.random.normal(size=1, loc=0, scale=np.sqrt(a3*delta_trans + a4*(abs(delta_rot1) + abs(delta_rot2))))
        cap_delta_rot2 = delta_rot2 + np.random.normal(size=1, loc=0, scale=np.sqrt(a1*abs(delta_rot2) + a2*delta_trans))

        p['x'] = p['x'] + cap_delta_trans[0] * np.cos(p['theta'] + cap_delta_rot1[0])
        p['y'] = p['y'] + cap_delta_trans[0] * np.sin(p['theta'] + cap_delta_rot1[0])
        p['theta'] = p['theta'] + cap_delta_rot1[0] + cap_delta_rot2[0]
        p['theta'] = normalize_angle(p['theta'])

    return particles


def measurement_model(particle, landmark):
    #Compute the expected measurement for a landmark
    #and the Jacobian with respect to the landmark.

    px = particle['x']
    py = particle['y']
    ptheta = particle['theta']

    lx = landmark['mu'][0]
    ly = landmark['mu'][1]

    #calculate expected range measurement
    meas_range_exp = np.sqrt((lx - px)**2 + (ly - py)**2)
    meas_bearing_exp = math.atan2(ly - py, lx - px) - ptheta
    meas_bearing_exp = normalize_angle(meas_bearing_exp)

    h = np.array([meas_range_exp, meas_bearing_exp])

    # Compute the Jacobian H of the measurement function h 
    #wrt the landmark location
    
    H = np.zeros((2,2))
    H[0,0] = (lx - px) / h[0]
    H[0,1] = (ly - py) / h[0]
    H[1,0] = (py - ly) / (h[0]**2)
    H[1,1] = (lx - px) / (h[0]**2)

    return h, H

def eval_sensor_model(sensor_data, particles):
    #Correct landmark poses with a measurement and
    #calculate particle weight

    #sensor noise
    Q_t = np.array([[0.1, 0],
                    [0, 0.1]])

    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearings = sensor_data['bearing']

    #update landmarks and calculate weight for each particle
    for p in particles:
        landmarks = p['landmarks']

        #loop over observed landmarks 
        for i in range(len(ids)):

            #current landmark
            lm_id = ids[i]
            landmark = landmarks[lm_id]
            
            #measured range and bearing to current landmark
            meas_range = ranges[i]
            meas_bearing = bearings[i]

            if not landmark['observed']:
                # landmark is observed for the first time
                
                # initialize landmark mean and covariance. You can use the
                # provided function 'measurement_model' above
                landmark['mu'] = np.array([p['x'] + meas_range * np.cos(normalize_angle(meas_bearing + p['theta'])),
                                           p['y'] + meas_range * np.sin(normalize_angle(meas_bearing + p['theta']))])
                _, H = measurement_model(p, landmark)
                inv_H = np.linalg.inv(H)
                landmark['sigma'] = np.dot(np.dot(inv_H, Q_t), inv_H.T)
                landmark['observed'] = True

            else:
                # landmark was observed before

                # update landmark mean and covariance. You can use the
                # provided function 'measurement_model' above. 
                # calculate particle weight: particle['weight'] = ...
                cap_zk, H = measurement_model(p, landmark)
                zt = np.array([meas_range, meas_bearing])
                difference = zt - cap_zk
                difference[1] = normalize_angle(difference[1])
                Sigma = landmark['sigma']
                Q = np.dot(np.dot(H, Sigma), H.T) + Q_t
                K = np.dot(np.dot(Sigma, H.T), np.linalg.inv(Q))
                landmark['mu'] = landmark['mu'] + np.dot(K, difference)
                landmark['sigma'] = np.dot((np.eye(2) - np.dot(K, H)), Sigma)

                p['weight'] *= multivariate_normal.pdf(zt, mean=cap_zk, cov=Q)


    #normalize weights
    normalizer = sum([p['weight'] for p in particles])

    for p in particles:
        p['weight'] = p['weight'] / normalizer

    return particles

def resample_particles(particles):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle 
    # weights.
    N = len(particles)
    weights = np.array([p['weight'] for p in particles])
    indices = np.random.choice(np.arange(N), size=N, p=weights)

    new_particles = [particles[i].copy() for i in indices]
    return new_particles

def predict_position(particles):
    pred_x, pred_y = 0, 0
    for p in particles:
        w = p['weight']
        pred_x += p['x'] * w
        pred_y += p['y'] * w

    return np.array([pred_x, pred_y])


def main():

    print("Reading landmark positions")
    landmarks = read_world("landmarks.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data("sensor_data.dat")

    num_particles = 200
    num_landmarks = len(landmarks)

    #create particle set
    particles = initialize_particles(num_particles, num_landmarks)
    estimated_positions = []
    #run FastSLAM
    for timestep in range(len(sensor_readings)//2):

        #predict particles by sampling from motion model with odometry info
        particles = sample_motion_model(sensor_readings[timestep,'odometry'], particles)

        #evaluate sensor model to update landmarks and calculate particle weights
        particles = eval_sensor_model(sensor_readings[timestep, 'sensor'], particles)

        estimated_positions.append(predict_position(particles))

        #calculate new set of equally weighted particles
        particles = resample_particles(particles)

    positions = np.array(estimated_positions)
    # Создаем график
    plt.figure(figsize=(10, 8))

    # 1. Траектория робота
    plt.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.7, label='Robot path')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.tight_layout()
    plt.show()

    normalizer = sum([p['weight'] for p in particles])
    for p in particles:
        p['weight'] = p['weight'] / normalizer

    pred_landmarks = np.array([np.array([0.0, 0.0]) for _ in range(num_landmarks)])
    for i in range(num_landmarks):
        for p in particles:
            pred_landmarks[i] += p['landmarks'][i + 1]['mu'] * p['weight']

    for i in range(num_landmarks):
        print(landmarks[i+1], pred_landmarks[i])


if __name__ == "__main__":
    main()