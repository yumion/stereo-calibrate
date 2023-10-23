import os
from datetime import datetime

import cv2
import numpy as np

from stereo_calibrate import StereoPreCalibration
from stereo_shoot import StereoCalibration


def main():
    stereo_camera = StereoCameraCapture(
        l_device="test_left.mp4",
        r_device="test_right.mp4",
        width=1024,
        height=576,
        fps=30,
        pattern_size=(7, 11),
        calib_data_filename="stereo_calibration_data_4x7.json",
    )
    stereo_camera.main_loop()


class StereoCameraCapture:
    def __init__(self, l_device, r_device, width, height, fps, pattern_size, calib_data_filename):
        self.l_device = l_device
        self.r_device = r_device
        self.width = width
        self.height = height
        self.fps = fps

        self.pre_calibrator = StereoPreCalibration(pattern_size, calib_data_filename)
        self.stereo_calibrator = StereoCalibration(l_device, r_device, width, height, fps)
        self.writerl, self.writerr = self.initialize_writers()
        self.capl, self.capr = self.stereo_calibrator.capl, self.stereo_calibrator.capr

        self.calibrated = False
        self.num_calibration = 1

    def initialize_writers(self):
        os.makedirs("videos/left", exist_ok=True)
        os.makedirs("videos/right", exist_ok=True)
        os.makedirs("images", exist_ok=True)
        left_filepath = f"left/{datetime.now().strftime('%Y%m%d_%H%M%S')}_left.mp4"
        right_filepath = f"right/{datetime.now().strftime('%Y%m%d_%H%M%S')}_right.mp4"
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        writerl = cv2.VideoWriter(left_filepath, codec, self.fps, (self.width, self.height))
        writerr = cv2.VideoWriter(right_filepath, codec, self.fps, (self.width, self.height))
        return writerl, writerr

    def main_loop(self):
        image_sets = []
        while True:
            retl, imgl = self.capl.read()
            retr, imgr = self.capr.read()
            if not retl or not retr:
                print("No more frames")
                break

            self.writerl.write(imgl)
            self.writerr.write(imgr)

            # esc/qが押された場合は終了する
            key = cv2.waitKey(1)
            if key == 27 or key == ord("q"):
                break

            # sが押された場合は保存する
            if key == ord("s"):
                print("Captured")
                now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"images/{now_str}_left.png", imgl)
                cv2.imwrite(f"images/{now_str}_right.png", imgr)
                image_sets.append([imgl, imgr])

            # cが押されたらキャリブレーションを開始する
            if key == ord("c") and len(image_sets) >= self.num_calibration:
                print("Start calibration")
                image_sets = np.array(image_sets)
                self.pre_calibrator.calibrate_camera(image_sets[:, 0], "l")
                self.pre_calibrator.calibrate_camera(image_sets[:, 1], "r")
                self.pre_calibrator.stereo_calibrate(image_sets)
                self.pre_calibrator.write_calibration_data()

                self.stereo_calibrator.calibration_data = self.pre_calibrator.calibration_data
                self.stereo_calibrator.load_calibration_data()
                self.stereo_calibrator.get_rectify_map()

                self.calibrated = True

            if self.calibrated:
                displ, dispr = self.stereo_calibrator.compute_disparity_map(imgl, imgr)
                filtered_img = self.stereo_calibrator.filter_disparity_map(imgl, displ, dispr)

                _3d_image = cv2.reprojectImageTo3D(filtered_img, self.stereo_calibrator.Q)

            if not self.calibrated:
                cv2.imshow("Left | Right", np.hstack([imgl, imgr]))
            else:
                cv2.imshow(
                    "Left | Right | Disparity",
                    np.hstack([imgl, imgr, cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)]),
                )

        self.capl.release()
        self.capr.release()
        self.writerl.release()
        self.writerr.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
