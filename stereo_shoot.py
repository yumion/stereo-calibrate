import json

import cv2
import numpy as np


def main():
    stereo_camera = StereoCalibration(
        l_device="videos/left/20231017_170153_left.mp4",
        r_device="videos/right/20231017_170152_right.mp4",
        calib_data_filename="stereo_calibration_data.json",
        width=640,
        height=480,
        fps=30,
    )
    stereo_camera.main_loop()


class StereoCalibration:
    def __init__(self, l_device, r_device, width, height, fps, calib_data_filename=None):
        self.l_device = l_device
        self.r_device = r_device
        self.width = width
        self.height = height
        self.fps = fps

        self.calib_data_filename = calib_data_filename
        self.calibration_data = {}
        self.capl, self.capr = self.initialize_cameras()
        # wsize default 3; 5; 7 for SGBM reduced size image;
        # 15 for SGBM full size. image (1300px and above); 5 Works nicely
        self.window_size = 11
        self.min_disp = 4
        self.num_disp = 128  # max_disp has to be dividable by 16 f. E. HH 192, 256
        self.left_matcher, self.right_matcher, self.wls_filter = self.initialize_matchers()

    def load_calibration_data(self):
        print("Load calibration data")
        if self.calib_data_filename is not None:
            with open(self.calib_data_filename, "r") as fr:
                self.calibration_data = json.load(fr)

        self.cameramatrixl = np.array(self.calibration_data["cameramatrixl"])
        self.cameramatrixr = np.array(self.calibration_data["cameramatrixr"])
        self.distcoeffsl = np.array(self.calibration_data["distcoeffsl"])
        self.distcoeffsr = np.array(self.calibration_data["distcoeffsr"])
        self.R = np.array(self.calibration_data["R"])
        self.T = np.array(self.calibration_data["T"])

        print("End load calibration data")

    def initialize_cameras(self):
        print("Open Cameras")
        capl = cv2.VideoCapture(self.l_device)
        capr = cv2.VideoCapture(self.r_device)

        capl.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capl.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capl.set(cv2.CAP_PROP_FPS, self.fps)
        capr.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capr.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capr.set(cv2.CAP_PROP_FPS, self.fps)

        return capl, capr

    def initialize_matchers(self):
        left_matcher = cv2.StereoSGBM_create(
            minDisparity=self.min_disp,  # 視差の下限
            numDisparities=self.num_disp,  # 視差の上限
            blockSize=self.window_size,  # 窓サイズ 3..11
            P1=8 * 3 * self.window_size**2,  # 視差の滑らかさを制御するパラメータ1
            P2=32 * 3 * self.window_size**2,  # 視差の滑らかさを制御するパラメータ2
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=50,  # 視差の滑らかさの最大サイズ. 50-100
            speckleRange=1,  # 視差の最大変化量. 1 or 2
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(80000)
        wls_filter.setSigmaColor(1.2)
        return left_matcher, right_matcher, wls_filter

    def get_rectify_map(self):
        flags = 0
        alpha = 1
        self.R1, self.R2, self.P1, self.P2, self.Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            self.cameramatrixl,
            self.distcoeffsl,
            self.cameramatrixr,
            self.distcoeffsr,
            (self.width, self.height),
            self.R,
            self.T,
            flags,
            alpha,
            (self.width, self.height),
        )

        m1type = cv2.CV_32FC1
        self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(
            self.cameramatrixl,
            self.distcoeffsl,
            self.R1,
            self.P1,
            (self.width, self.height),
            m1type,
        )
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(
            self.cameramatrixr,
            self.distcoeffsr,
            self.R2,
            self.P2,
            (self.width, self.height),
            m1type,
        )

    def compute_disparity_map(self, imgl, imgr):
        imgl_gray = cv2.cvtColor(imgl, cv2.COLOR_BGR2GRAY)
        imgr_gray = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)

        imgl_gray = cv2.remap(imgl_gray, self.map1_l, self.map2_l, cv2.INTER_NEAREST)
        imgr_gray = cv2.remap(imgr_gray, self.map1_r, self.map2_r, cv2.INTER_NEAREST)

        displ = self.left_matcher.compute(imgl_gray, imgr_gray)
        dispr = self.right_matcher.compute(imgr_gray, imgl_gray)
        displ = np.int16(displ)
        dispr = np.int16(dispr)

        return displ, dispr

    def filter_disparity_map(self, imgl, displ, dispr):
        filtered_img = self.wls_filter.filter(displ, imgl, None, dispr)
        filtered_img = cv2.normalize(
            src=filtered_img, dst=filtered_img, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX
        )
        filtered_img = np.uint8(filtered_img)
        return filtered_img

    def main_loop(self):
        self.load_calibration_data()
        self.get_rectify_map()
        while True:
            retl, imgl = self.capl.read()
            retr, imgr = self.capr.read()
            if not retl or not retr:
                print("No more frames")
                break

            displ, dispr = self.compute_disparity_map(imgl, imgr)
            filtered_img = self.filter_disparity_map(imgl, displ, dispr)

            _3d_image = cv2.reprojectImageTo3D(filtered_img, self.Q)

            cv2.imshow(
                "Stereo",
                cv2.hconcat([imgl, imgr, cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)]),
            )

            k = cv2.waitKey(1)
            if k == 27:
                break

        self.capl.release()
        self.capr.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
