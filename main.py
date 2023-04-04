from Grid import Grid
import cv2 as cv
import numpy as np
from tqdm import tqdm


def main():
    fps = 1
    img_layers = fps * 2
    xmin = -img_layers - 1
    ymin = -img_layers - 1
    xmax = -1 * xmin
    ymax = -1 * ymin
    h = 480
    w = 480

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    video = cv.VideoWriter("outvid.avi", fourcc, fps, (w, h))
    for _ in tqdm(range(img_layers)):
        grid = Grid(xmin, xmax, ymin, ymax, (h, w))
        frame = grid.gen_image()

        cvFrame = np.array(frame)
        cvFrame = cvFrame[:, :, ::-1].copy()

        video.write(cvFrame)

        xmin += 1
        ymin += 1
        xmax = -1 * xmin
        ymax = -1 * ymin

    cv.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    main()
