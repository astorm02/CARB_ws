import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import math

## Functions for quaternion and rotation matrix conversion
## The code is adapted from the general_robotics_toolbox package
## Code reference: https://github.com/rpiRobotics/rpi_general_robotics_toolbox_py
def hat(k):
    """
    Returns a 3 x 3 cross product matrix for a 3 x 1 vector

             [  0 -k3  k2]
     khat =  [ k3   0 -k1]
             [-k2  k1   0]

    :type    k: numpy.array
    :param   k: 3 x 1 vector
    :rtype:  numpy.array
    :return: the 3 x 3 cross product matrix
    """

    khat=np.zeros((3,3))
    khat[0,1]=-k[2]
    khat[0,2]=k[1]
    khat[1,0]=k[2]
    khat[1,2]=-k[0]
    khat[2,0]=-k[1]
    khat[2,1]=k[0]
    return khat

def q2R(q):
    """
    Converts a quaternion into a 3 x 3 rotation matrix according to the
    Euler-Rodrigues formula.
    
    :type    q: numpy.array
    :param   q: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :rtype:  numpy.array
    :return: the 3x3 rotation matrix    
    """
    
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2*q[0]*qhat + 2*qhat2
######################

def euler_from_quaternion(q):
    w=q[0]
    x=q[1]
    y=q[2]
    z=q[3]
    # euler from quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - z * x))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    return [roll,pitch,yaw]

class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.get_logger().info('Tracking Node Started')
        
        # Current object pose
        self.obs_pose = None
        self.goal_pose = None
        self.prev_obs_pose = None
        
        # ROS parameters
        self.declare_parameter('world_frame_id', 'odom')

        # Create a transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Create publisher for the control command
        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)
        # Create a subscriber to the detected object pose
        self.sub_detected_goal_pose = self.create_subscription(PoseStamped, 'detected_color_object_pose', self.detected_obs_pose_callback, 10)
        self.sub_detected_obs_pose = self.create_subscription(PoseStamped, 'detected_color_goal_pose', self.detected_goal_pose_callback, 10)

        # Create timer, running at 100Hz
        self.timer = self.create_timer(0.01, self.timer_update)
    
    def detected_obs_pose_callback(self, msg):
        #self.get_logger().info('Received Detected Object Pose')
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        # TODO: Filtering
        # You can decide to filter the detected object pose here
        # For example, you can filter the pose based on the distance from the camera
        # or the height of the object
        
        if np.linalg.norm(center_points) > 4 or center_points[2] > 0.7:
            return
        

        try:
            # Transform the center point from the camera frame to the world frame
            transform = self.tf_buffer.lookup_transform(odom_id,msg.header.frame_id,rclpy.time.Time(),rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w,transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z]))
            cp_world = t_R@center_points+np.array([transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return

        
        # Get the detected object pose in the world frame
        self.obs_pose = cp_world

    def detected_goal_pose_callback(self, msg):
        #self.get_logger().info('Received Detected Object Pose')
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        # TODO: Filtering
        # You can decide to filter the detected object pose here
        # For example, you can filter the pose based on the distance from the camera
        # or the height of the object

        if np.linalg.norm(center_points) > 4 or center_points[2] > 0.7:
            return
        
        try:
            # Transform the center point from the camera frame to the world frame
            transform = self.tf_buffer.lookup_transform(odom_id,msg.header.frame_id,rclpy.time.Time(),rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w,transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z]))
            cp_world = t_R@center_points+np.array([transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return
        
        # Get the detected object pose in the world frame
        self.goal_pose = cp_world
        
    def get_current_poses(self):
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        # Get the current robot pose
        try:
            # from base_footprint to odom
            transform = self.tf_buffer.lookup_transform('base_footprint', odom_id, rclpy.time.Time())
            robot_world_x = transform.transform.translation.x
            robot_world_y = transform.transform.translation.y
            robot_world_z = transform.transform.translation.z
            robot_world_R = q2R([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z])
            obstacle_pose = robot_world_R@self.obs_pose+np.array([robot_world_x,robot_world_y,robot_world_z])
            goal_pose = robot_world_R@self.goal_pose+np.array([robot_world_x,robot_world_y,robot_world_z])
    
        
        except TransformException as e:
            self.get_logger().error('Transform error: ' + str(e))
            return
        
        return obstacle_pose, goal_pose
    
    def timer_update(self):

        if self.goal_pose is None or self.obs_pose is None:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.pub_control_cmd.publish(cmd_vel)
            # self.get_logger().error('Objects not Found')
            return
            
        # print("successfully found poses")
        current_obs_pose, current_goal_pose = self.get_current_poses()
        # print("Obstacle Pose:")
        # print(current_obs_pose)
        # print("Goal Pose:")
        # print(current_goal_pose)
        
        # print('Running Controller Algorithm')
        cmd_vel = self.controller(current_obs_pose, current_goal_pose)
        # print("Publishing Controller Command")
        self.pub_control_cmd.publish(cmd_vel)
        
        return
    
    def controller(self, current_obs_pose, current_goal_pose):
        # Instructions: You can implement your own control algorithm here
        # feel free to modify the code structure, add more parameters, more input variables for the function, etc.
        cmd_vel = Twist()
        
        ########### Write your code here ###########
        
        # Goal Reached Threshold
        goal_thresh_dist = 0.4

        # Determine Distance to the Goal
        goal_vec = current_goal_pose[:2]
        goal_dist = np.linalg.norm(goal_vec)
        unit_goal_vec = goal_vec/goal_dist # normalizes the vector to the goal
        #print(f"Goal Distance:{goal_dist}")

        # End if goal is reached
        if goal_dist < goal_thresh_dist:
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            return cmd_vel
        
        # Potential Field Gains
        k_atr = 4
        k_rep = 3

        # Obstacle Avoidance Distance
        obs_avoid_dist = 0.6

        # Determine Attractive Froce
        F_atr = k_atr*unit_goal_vec
        #print(f"Attractive:{F_atr}")

        # Determine Repulsive Force


        if current_obs_pose is not None and self.prev_obs_pose is not None:
            
            if self.prev_obs_pose is not current_obs_pose:

                obs_vec = current_obs_pose[:2]
                obs_dist = np.linalg.norm(obs_vec)
                #print(f"Obstacle Distance:{obs_dist}")
                unit_obs_vec = obs_vec/obs_dist # normalizes the vector to the obstacle
                self.prev_obs_pose = current_obs_pose

                if obs_dist < obs_avoid_dist:

                    F_rep = k_rep*((1.0/obs_dist)-(1.0/obs_avoid_dist))*(1.0/(obs_dist**2))*(-unit_obs_vec)

                else:

                    F_rep = np.array([0.0,0.0])

            else:
                
                F_rep = np.array([0.0,0.0])

        else:

            self.prev_obs_pose = np.array([0.0,0.0,0.0])
            F_rep = np.array([0.0,0.0])
        
        

        #print(f"Repulsive:{F_rep}")
        # Determine Total Force
        F_tot = F_atr + F_rep

        u = F_tot/np.linalg.norm(F_tot)
        print(u)

        cmd_vel.linear.x = u[0]
        cmd_vel.linear.y = u[1]
        

        return cmd_vel
        
def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)
    # Create the node
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    # Destroy the node explicitly
    tracking_node.destroy_node()
    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()
