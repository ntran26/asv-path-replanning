import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class RadarProcess:
    def __init__(self, path_to_data, data="new_data.png", nm_range=1/16):
        """
        Initialize data path, image, scanning range

        Args:
            path_to_data: directory of data
            data: name of data
            nm_range: maximum scanning range in nautical miles
        """
        self.path_to_data = path_to_data
        self.data = data
        self.nm_range = nm_range
        self.meter_range = nm_range * 1852  # convert nautical miles to meters
        self.polar_data = None
        self.thresh = None
        self.ranges_meter = None

    def load_and_process(self):
        img_path = os.path.join(self.path_to_data, self.data)
        self.polar_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if self.polar_data is None:
            raise FileNotFoundError(f"Cannot find image at {img_path}")

        # apply gaussian and median filters
        gaussian = cv2.GaussianBlur(self.polar_data, (5, 5), 0)
        median = cv2.medianBlur(gaussian, 5)

        # convert to binary
        _, self.thresh = cv2.threshold(median, 50, 255, cv2.THRESH_BINARY)

        return self.thresh
    
    def compute_ranges(self, start_col=8):
        ranges = []
        radar = self.thresh[:,:]
        radar = np.flipud(radar)
        angle, distance = radar.shape

        for i in range(angle):
            for j in range(start_col,distance):     # ignore radar size from the raw image
                if radar[i,j] > 0:
                    ranges.append(j+start_col)      # compensate for radar size            
                    break
            else:
                ranges.append(distance)

        ranges = np.array(ranges)
        self.ranges_meter = np.round((ranges*self.meter_range)/distance, 2)
        return self.ranges_meter
    
    def process(self):
        self.load_and_process()
        return self.compute_ranges()
    
if __name__ == "__main__":
    processor = RadarProcess(path_to_data="radar/sample_data", data="new_data.png")
    ranges_meter = processor.process()
    print(ranges_meter)

# path_to_data = "radar/sample_data"

# # nautical miles to meter
# nm_range = 1/16
# meter_range = nm_range * 1852

# # fetch images and convert to grayscale
# polar_data = cv2.imread(os.path.join(path_to_data,"polar_scan.png"), cv2.IMREAD_GRAYSCALE)
# top_data = cv2.imread(os.path.join(path_to_data,"top_scan.png"), cv2.IMREAD_GRAYSCALE)
# polar_data = cv2.imread(os.path.join(path_to_data,"new_data.png"), cv2.IMREAD_GRAYSCALE)

# # apply gaussian and median filtering, and thresholding
# gaussian = cv2.GaussianBlur(polar_data, (5, 5), 0)
# median = cv2.medianBlur(gaussian, 5)
# _, thresh = cv2.threshold(median, 50, 255, cv2.THRESH_BINARY)
# # print(thresh[100,:])

# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(polar_data, cmap="gray")
# axes[0].set_title("Raw Image")
# axes[1].imshow(thresh, cmap="gray")
# axes[1].set_title("Filtered Image")
# plt.tight_layout()
# plt.show()

# # set up lidar ranges
# # scan through rows and columns (angles and distances)
# # find the closest obstacle in each row (angle)
# ranges = []
# radar = thresh[:,:]
# angle, distance = radar.shape

# for i in range(angle):
#     for j in range(8,distance):
#         if radar[i,j] > 0:
#             ranges.append(j+8)
#             break
#     else:
#         ranges.append(distance)

# ranges = np.array(ranges)
# ranges_meter = np.round((ranges*meter_range)/distance, 2)
# print(ranges_meter)

# # cv2.imwrite('radar/filtered.png',radar)