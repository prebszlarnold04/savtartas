import cv2
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32
#from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
from rclpy.node import Node
import rclpy
from filterpy.kalman import KalmanFilter

class LaneDetect(Node):
    def __init__(self):
        super().__init__('lane_detect')
        # parameters
        self.previous_center = None  # Előző középpont
        self.declare_parameter('raw_image', False)
        self.declare_parameter('image_topic', '/image_raw')
        img_topic = self.get_parameter('image_topic').value
        if self.get_parameter('raw_image').value:
            self.sub1 = self.create_subscription(Image, img_topic, self.raw_listener, 10)
            self.sub1  # prevent unused variable warning
            self.get_logger().info(f'lane_detect subscribed to raw image topic: {img_topic}')
        else:
            self.sub2 = self.create_subscription(CompressedImage, '/image_raw/compressed', self.compr_listener, 10)
            self.sub2  # prevent unused variable warning
            self.get_logger().info(f'lane_detect subscribed to compressed image topic: {img_topic}')
        self.pub1 = self.create_publisher(Image, '/lane_img', 10)
        self.bridge = CvBridge()
        self.center_pub = self.create_publisher(Float32, '/lane_offset', 10)
        self.previous_center = None

        self.kf = KalmanFilter(dim_x=2, dim_z=1)  # 2 állapotváltozó (helyzet, sebesség)
        self.kf.x = np.array([0, 0])  # Kezdeti állapot: [középpont, sebesség]
        self.kf.F = np.array([[1, 1], [0, 1]])  # Átmeneti mátrix
        self.kf.H = np.array([[1, 0]])  # Megfigyelési mátrix
        self.kf.P *= 1000  # Nagy kezdeti bizonytalanság
        self.kf.R = 10 # Mérési zaj (állítsd be az érzékenységhez) Nagyobb érték → kevésbé bízunk a mérésekben (simább, lassabb)
        self.kf.Q = np.array([[5, 0], [0, 5]])  # Folyamat zaj (állítsd be az érzékenységhez) Nagyobb érték → gyorsabban reagál, de instabilabb lehet

        self.previous_center = None  # Előző középpont
        

    def raw_listener(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # print info of the image, only once not every time
        self.get_logger().info(f'First raw img arrived, shape: {cv_image.shape}', once=True)
        # Detect lanes
        lane_image = self.detect_lanes(cv_image) 
        # Convert OpenCV image to ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(lane_image, 'bgr8')
        # Publish the image
        self.pub1.publish(ros_image)
    
    def compr_listener(self, msg):
        # Convert ROS CompressedImage message to OpenCV image
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # print info of the image, only once not every time
        self.get_logger().info(f'First compressed img arrived, shape: {cv_image.shape}', once=True)
        # Detect lanes
        lane_image = self.detect_lanes(cv_image)
        # Convert OpenCV image to ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(lane_image, 'bgr8')
        # Publish the image
        self.pub1.publish(ros_image)

    

    def detect_lanes(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Detect edges using Canny
        edges = cv2.Canny(blur, 50, 150)
        # Define a region of interest
        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height),
            (width, height),
            (width, int(height * 0.6)),        # nemet mcap-hez: 0.45   f1tenth-hez: 0.6
            (0, int(height * 0.6))              # nemet mcap-hez 0.45   f1tenth-hez: 0.6
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, 50, maxLineGap=50)
        line_image = np.zeros_like(image)
        left_x = []
        right_x = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > 50:  # Csak a 30 pixelnél hosszabb vonalakat tartjuk meg
                    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
                    if abs(slope) > 0.1:  # Meredekebb vonalakat fogadunk csak el
                        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        if x1 < width / 2 and x2 < width / 2:
                            left_x.extend([x1, x2])
                        elif x1 > width / 2 and x2 > width / 2:
                            right_x.extend([x1, x2])

        # Calculate the center with fallback to default if one side is missing
        center = width / 2  # Default center
        width_of_road = None
        count = 0

        if left_x and right_x:
            if count == 0:
                width_of_road = np.mean(right_x) - np.mean(left_x)
                count += 1
            else:
                count += 1
                width_of_road += (np.mean(right_x) - np.mean(left_x)) / count
            left_avg = np.mean(left_x)
            right_avg = np.mean(right_x)
            center = (left_avg + right_avg) / 2
        elif left_x:
            left_avg = np.mean(left_x)
            if width_of_road != None:
                right_avg = left_avg + width_of_road
                center = (left_avg + right_avg) / 2
            else:
                center = (left_avg + (width * 2)) / (2 + (1-abs(slope)))
            #center = left_avg + np.mean(width * (1 - abs(slope)))
        elif right_x:
            #left_avg = 0
            right_avg = np.mean(right_x)
            if width_of_road != None:
                left_avg = right_avg - width_of_road
                center = (left_avg + right_avg) / 2
            else:
                center = right_avg / (2 + (1-abs(slope)))
            #center = right_avg - np.mean(width * (1 - abs(slope)))


        
        
        # Kalman-szűrő alkalmazása
        self.kf.predict()  # Előrejelzés
        self.kf.update(np.array([center]))  # Frissítés az új méréssel
        center = self.kf.x[0]  # A szűrt középpontot kapjuk meg


        # Draw the center on the image
        cv2.line(line_image, (int(center), height), (int(center), int(height * 0.6)), (255, 0, 0), 2)
        cv2.circle(line_image, (int(center), int(height / 2)), 10, (255, 0, 0), -1)

        # Combine the original image with the line image
        combined_image = cv2.addWeighted(image, 0.25, line_image, 1, 1)

        # Publish the center as a Float32 message
        center_msg = Float32()
        center_msg.data = center
        self.center_pub.publish(center_msg)

        return combined_image

  

    

        
        
    
def main(args=None):
    rclpy.init(args=args)
    lane_detect = LaneDetect()
    rclpy.spin(lane_detect)
    lane_detect.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()