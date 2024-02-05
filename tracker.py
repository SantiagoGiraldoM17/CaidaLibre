import numpy as np 
import cv2
import matplotlib.pyplot as plt
import IPython.display as ipd

class ScaleSetter:
    def __init__(self, image):
        self.ref_points = []
        self.image = image.copy()  # Make a copy of the image to draw on
        self.scale_cm_per_pixel = None

    def click_and_mark(self, event, x, y, flags, param):
        """Mouse callback function to mark points and draw the line."""
        if event == cv2.EVENT_LBUTTONUP:
            self.ref_points.append((x, y))
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)

            if len(self.ref_points) == 2:
                cv2.line(self.image, self.ref_points[0], self.ref_points[1], (0, 255, 0), 2)
                pixel_distance = np.linalg.norm(np.array(self.ref_points[0]) - np.array(self.ref_points[1]))
                self.scale_cm_per_pixel = 1 / pixel_distance  # Assuming the distance represents 1cm

    def set_scale(self):
        """Invoke this method to set the scale by manually marking two points on the image."""
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.click_and_mark)

        while True:
            cv2.imshow("Image", self.image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or len(self.ref_points) == 2:  # Quit or two points have been marked
                break

        cv2.destroyAllWindows()
        if self.scale_cm_per_pixel is not None:
            print(f"Scale set: {self.scale_cm_per_pixel} cm/pixel")
        else:
            print("Scale not set. Ensure two points were marked.")

        return self.scale_cm_per_pixel

def drawBox(img, bbox):
    x,y,w,h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x,y), ((x+w),(y+h)),(255,0,255),3,1)

cap = cv2.VideoCapture("CaidaLibre.mp4")
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

tracker = cv2.legacy.TrackerMOSSE.create()
ret, img = cap.read()
bbox = cv2.selectROI("Tracking", img, False)
tracker.init(img,bbox)

scale_setter = ScaleSetter(img)

scale_cm_per_pixel = scale_setter.set_scale()

fig, axs = plt.subplots(3, 9, figsize=(15,30))
axs = axs.flatten ()
img_idx = 0

list_y = []
list_frames = []
for frame in range(frames):
    ret, img = cap.read()
    if ret == False:
        break
    ret1, bbox = tracker.update(img)
    if ret1:
        if frame % 10 == 0:
            drawBox(img, bbox)
            axs[img_idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[img_idx].set_title(f'Frame: {frame}')
            axs[img_idx].axis('off')
            img_idx += 1
            y = int(bbox[1])
            list_y.append(y)
            list_frames.append(frame)
        

times = np.array([list_frames[i] / 1000 for i in range(len(list_y))])
distances_pixels = np.array([list_y[i] for i in range(len(list_y))])  # Distance in pixels
distances_mm = ((distances_pixels - distances_pixels[0]) * scale_cm_per_pixel * 10)

time_differences = np.diff(times)
distance_differences = np.diff(distances_mm)
speeds_mm_per_s = distance_differences / time_differences

distances_m = distances_mm / 1000
speeds_m_per_s = speeds_mm_per_s / 1000

# Plot Distance vs. Time
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(times, distances_m, '-o')
plt.title('Distance vs. Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Distance (meters)')
plt.grid(True)

# Plot Speed vs. Time
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(times[1:], speeds_m_per_s, '-o')  # Exclude the first time point as speeds array is one less
plt.title('Speed vs. Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Speed (m/s)')
plt.grid(True)

plt.tight_layout()
plt.show()



