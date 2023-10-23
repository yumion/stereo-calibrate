import cv2
from tqdm import trange

codec = cv2.VideoWriter_fourcc(*"mp4v")
writerl = cv2.VideoWriter("test_left.mp4", codec, 30.0, (1024, 576))
writerr = cv2.VideoWriter("test_right.mp4", codec, 30.0, (1024, 576))

for _ in trange(1000 // 20):
    for i in range(20):
        left_img = cv2.imread(f"data/imgs/leftcamera/Im_L_{i + 1}.png")
        writerl.write(left_img)
        right_img = cv2.imread(f"data/imgs/rightcamera/Im_R_{i + 1}.png")
        writerr.write(right_img)

writerl.release()
writerr.release()
