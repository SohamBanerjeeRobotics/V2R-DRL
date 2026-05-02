# Importing necessary Gymnasium Libraries
import gymnasium as gym
from gymnasium import spaces
# Importing Pybullet dependencies
import pybullet as p
import pybullet_data
# Importing other libraries
import numpy as np

class TrainingEnvOne(gym.Env):
    def __init__(self, render=True):
        super().__init__()
        
        self.render_mode = render
	
	# ----------------------- Observation Space ------------------------------------------------
	self.observation_space = spaces.Box(low = np.array([0.0] * 10 + [0.0, -np.pi] + [0.0, -1.0]), high = np.array([1.0] * 10 + [np.inf, np.pi] + [1.0, 1.0]), dtype = np.float32)
	
	# ----------------------- Action Space -----------------------------------------------------
	self.action_space = spaces.Box(low = np.array([0.0, -1.0]), high = np.array([1.0, 1.0]), dtype = np.float32)
	
	# ----------------------- Pybullet ---------------------------------------------------------
	if self.render_mode:
	    p.connect(p.GUI) # Connects to the Pybullet GUI
	else:
	    p.connect(p.DIRECT) # Without GUI
	    
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # laser parameters
        self.num_rays = 10
        self.max_range = 2.0
        
        self.dt = 0.05 # 50 milisecods.
        
        
    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        p.loadURDF("plane.urdf")
        
        self.build_walls()
        self.build_obstacles()
        self.spawn_robot()
        self.spawn_target()
        
        self.prev_distance = None
        self.prev_velocity = np.array([0.0, 0.0])
        self.step_count = 0
        
        if self.render_mode:
            p.resetDebugVisualizerCamera(cameraDistance = 12, cameraYaw = 45, cameraPitch = -45, cameraTargetPosition = [0, 0, 0])
         
        return self.get_obs(), {}
        
        
    def step(self, action):
    	lin, ang = action
    	
    	# Scaling as per paper
    	lin = lin * 0.5  # max linear velocity 0.5 m/s
    	ang = ang * 1.0  # max angular velocity 1.0 m/s
    	
    	# get current pose
    	pos, orn = p.getBasePositionAndOrientation(self.robot)
    	yaw = p.getEulerFromQuaternion(orn)[2]
    	
    	# Differential Drive Motion
    	new_x = pos[0] + lin * np.cos(yaw) * self.dt
    	new_y = pos[1] + lin * np.sin(yaw) * self.dt
	new_yaw = yaw + ang * self.dt
	
	new_orn = p.getQuaternionFromEuler([0, 0, new_yaw])
	
	p.resetBasePositionAndOrientation(self.robot, [new_x, new_y, pos[2]], new_orn)
	
	p.stepSimulation()
	
	obs = self.get_obs()
	
	# ---------------------------- Reward --------------------------------------------
	
	reward, terminated = self.compute_reward()
	
	self.prev_velocity = action
	self.step_count += 1
	
	truncated = self.step_count > 500
	
	return obs, reward, terminated, truncated, {}
	
	
    def build_walls(self):
	wall_thickness = 0.1
	wall_length = 10
	wall_height = 1

	wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents = [wall_thickness/2, wall_length/2, wall_height/2])
	
	wall_visual = p.createVisualShape(p.GEOM_BOX,  halfExtents = [wall_thickness/2, wall_length/2, wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1])
	
	positions = [[wall_length/2 + 0.05, 0, wall_height/2], [-wall_length/2 - 0.05, 0, wall_height/2], [0, wall_length/2 + 0.05, wall_height/2], [0, -wall_length/2 - 0.05, wall_height/2]]
	
	orientations = [p.getQuaternionFromEuler([0, 0, 0]), p.getQuaternionFromEuler([0, 0, 0]), p.getQuaternionFromEuler([0, 0, np.pi/2]), p.getQuaternionFromEuler([0, 0, np.pi/2])]
	
	for pos, ori in zip(positions, orientations):
	    p.createMultiBody(baseMass = 0, baseCollisionShapeIndex = wall_shape, baseVisualShapeIndex = wall_visual, basePosition = pos, baseOrientation = ori)

    def build_obstacles(self):
    	obs_thickness = 0.50
	obs_length = 1
	obs_height = 0.25

	self.obstacle_ids = []	
		
	obs_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents = [obs_thickness/2, obs_length/2, obs_height/2])
	obs_visual = p.createVisualShape(p.GEOM_BOX,  halfExtents = [obs_thickness/2, obs_length/2, obs_height/2], rgbaColor=[0.3, 0.3, 0.3, 1])
	
	positions = [[2, 0, obs_height/2], [-1, -1, obs_height/2], [-0.5, 2, obs_height/2]]
	orientations = [p.getQuaternionFromEuler([0, 0, 0]), p.getQuaternionFromEuler([0, 0, 0.75]), p.getQuaternionFromEuler([0, 0, 1.57])]
	
	for pos, ori in zip(positions, orientations):
	    self.obstacle_ids.append(p.createMultiBody(baseMass = 0, baseCollisionShapeIndex = obs_shape, baseVisualShapeIndex = obs_visual, basePosition = pos, baseOrientation = ori))
	    
    def spawn_robot(self):
    	self.robot = p.loadURDF("r2d2.urdf", [0, 0, 0.1])
    
    def spawn_target(self):
    	while True:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            
            # Write code to check for collision, and if collision then don't break. 
            target_shape = p.createCollisionShape(p.GEOM_SPHERE, radius = 0.2)
            
            temp_target = p.createMultiBody(baseMass = 0, baseCollisionShapeIndex = target_shape, basePosition = [x, y, 0.2])
            
            collison = False
            
            for obs_id in self.obstacle_ids:
                pts = p.getClosestPoints(temp_target, obs_id, distance = 0.0)
                if len(pts) > 0:
                    collision = True
                    break
            
            p.removeBody(temp_target)
            
            if not collision:
                self.target = np.array([x, y])
                
                p.loadURDF("sphere2.urdf", [x, y, 0.2])
                break
            
    
    def get_obs(self):
        laser = self.get_laser()
        target = self.get_target_relative()
        
        return np.concatenate([laser, target, self.prev_velocity])
    
    def get_target_relative(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(orn)[2]
        
        dx = self.target[0] - pos[0]
        dy = self.target[1] - pos[1]
        
        dist = np.sqrt((dx ** 2) + (dy ** 2))
        angle = np.arctan2(dy, dx) - yaw
        
        return np.array([dist, angle])
        
        
    def get_laser(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(orn)[2]
        
        angles = np.linspace(-np.pi / 2, np.pi / 2, self.num_rays)
        ray_from = []
        ray_to = []
        
        for a in angles:
            angle = yaw + a
            ray_from.append(pos)
            ray_to.append([pos[0] + self.max_range * np.cos(angle), pos[1] + self.max_range * np.sin(angle), pos[2])
            
        results = p.rayTestBatch(ray_from, ray_to)
        
        distances = []
        for r in results:
            hit = r[2]
            dist = hit * self.max_range
            distances.append(dist / self.max_range)
        
        return np.array(distances)
        
        
    def compute_reward(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        robot_xy = np.array(pos[:2])
        
        # current dist
        dist = np.linalg.norm(robot_xy - self.target)
        
        # Goal
        if dist < 0.3:
            return 100.0, True
        
        # Collision
        laser = self.get_laser()
        if np.min(laser) < 0.1:
            return -100.0, True
            
        # Progress
        if self.prev_distance is None:
            self.prev_distance = dist
        reward = 1.0 * (self.prev_distance - dist)
        
        self.prev_distance = dist
        
        return reward, False       
