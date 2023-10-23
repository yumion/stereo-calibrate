import glob
import json

import cv2
import numpy as np


def main():
    """
    Main function to handle stereo camera calibration.
    """

    PATTERN_SIZE = (7, 11)
    CALIBRATION_DATA_FILENAME = "stereo_calibration_data.json"

    calibrator = StereoPreCalibration(PATTERN_SIZE, CALIBRATION_DATA_FILENAME)
    imagesl = sorted(glob.glob("data/imgs/leftcamera/*.png"))  # 左目
    imagesr = sorted(glob.glob("data/imgs/rightcamera/*.png"))  # 右目
    # 左目,右目のセット
    image_set = list(zip(imagesl, imagesr))

    calibrator.calibrate_camera(imagesl, "l")
    calibrator.calibrate_camera(imagesr, "r")
    calibrator.stereo_calibrate(image_set)
    calibrator.write_calibration_data()


class StereoPreCalibration:
    def __init__(self, pattern_size, calib_data_filename):
        self.pattern_size = pattern_size
        self.calib_data_filename = calib_data_filename
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
        self.calibration_data = {}

    def match_images(self, imagesl, imagesr):
        image_set = []
        for imagel in imagesl:
            print("matching... left image: " + imagel)
            basename = imagel[0 : -len(self.left_suffix)]
            if basename + self.right_suffix in imagesr:
                image_set.append((basename + self.left_suffix, basename + self.right_suffix))
                print("matched. ({}) <-> ({})".format(image_set[-1][0], image_set[-1][1]))
            else:
                print("matching fail.")
        return image_set

    def calibrate_camera(self, images, side):
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        print(f"[Info] Starting {side} camera calibration...")

        for image in images.copy():
            if type(image) == np.ndarray:
                img = image
            else:
                print(f"read image file {image}")
                img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
            if ret:
                objpoints.append(self.objp)
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                imgpoints.append(corners)
                cv2.drawChessboardCorners(img, self.pattern_size, corners, ret)
                cv2.imshow(f"img_{side}", img)
                cv2.waitKey(500)
        cv2.destroyAllWindows()

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (w, h), None, None
        )
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        self.calibration_data[f"mtx{side}"] = mtx
        self.calibration_data[f"dist{side}"] = dist
        self.calibration_data[f"newcameramtx{side}"] = newcameramtx
        print(f"mtx{side}: {mtx}")
        print(f"dist{side}: {dist}")
        print(f"newcameramtx{side}: {newcameramtx}")
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        cv2.imshow(f"calibresult_{side}", dst)
        cv2.waitKey(500)

    def stereo_calibrate(self, image_set):
        objpoints = []  # 3d point in real world space
        imgpointsl = []  # 2d points in image plane.
        imgpointsr = []  # 2d points in image plane.
        for image_lr in image_set.copy():
            if type(image_lr[0]) == np.ndarray:
                im_l = image_lr[0]
                im_r = image_lr[1]
            else:
                print(f"read image L file {image_lr[0]}")
                print(f"read image R file {image_lr[1]}")
                im_l = cv2.imread(image_lr[0])
                im_r = cv2.imread(image_lr[1])

            gray_l = cv2.cvtColor(im_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(im_r, cv2.COLOR_BGR2GRAY)
            found_l, corners_l = cv2.findChessboardCorners(gray_l, self.pattern_size, None)
            found_r, corners_r = cv2.findChessboardCorners(gray_r, self.pattern_size, None)
            if found_l and found_r:
                objpoints.append(self.objp)
                cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), self.criteria)
                cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), self.criteria)
                imgpointsl.append(corners_l)
                imgpointsr.append(corners_r)
                cv2.drawChessboardCorners(im_l, self.pattern_size, corners_l, found_l)
                cv2.drawChessboardCorners(im_r, self.pattern_size, corners_r, found_r)
                cv2.imshow("img L|R", np.hstack([im_l, im_r]))
                cv2.waitKey(500)
        (
            retval,
            cameramatrixl,
            distcoeffsl,
            cameramatrixr,
            distcoeffsr,
            R,
            T,
            E,
            F,
        ) = cv2.stereoCalibrate(
            objpoints,
            imgpointsl,
            imgpointsr,
            self.calibration_data["newcameramtxl"],
            self.calibration_data["distl"],
            self.calibration_data["newcameramtxr"],
            self.calibration_data["distr"],
            gray_l.shape[::-1],
        )
        self.calibration_data["cameramatrixl"] = cameramatrixl
        self.calibration_data["cameramatrixr"] = cameramatrixr
        self.calibration_data["distcoeffsl"] = distcoeffsl
        self.calibration_data["distcoeffsr"] = distcoeffsr
        self.calibration_data["R"] = R
        self.calibration_data["T"] = T
        print("cameramatrixl: {}".format(cameramatrixl))
        print("cameramatrixr: {}".format(cameramatrixr))
        print("distcoeffsl: {}".format(distcoeffsl))
        print("distcoeffsr: {}".format(distcoeffsr))
        print("R: {}".format(R))
        print("T: {}".format(T))
        cv2.waitKey(1000)

    def write_calibration_data(self):
        with open(self.calib_data_filename, "w") as fw:
            json.dump(self.calibration_data, fw, indent=4, cls=ExtendedJsonEncoder)
        print("Write Calibration Data in File {}".format(self.calib_data_filename))
        print("Completely task ended.")


class ExtendedJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(ExtendedJsonEncoder, self).default(obj)


if __name__ == "__main__":
    main()
