import os
import argparse
import multiprocessing
import logging
import time
import glob
import pydicom
import SimpleITK as sitk
import numpy as np
import cv2
import traceback


parser = argparse.ArgumentParser(description='Convert dicom to jpg')
parser.add_argument('dcm_path', default=None, metavar='DCM_PATH',
                    type=str, help="Path to the input dcm files")
parser.add_argument('jpg_path', default=None, metavar='JPG_PATH',
                    type=str, help="Path to the output jpg files")
parser.add_argument('--num_workers', default=1, type=int, help='num_workers')
parser.add_argument('--no_equalize_hist', action='store_false',
                    dest='equalize_hist', help="Do not equalize_hist raw "
                                               "images, if not set, the default value of equalize_hist "
                                               "is True")
parser.add_argument('--percent_min', default=1, type=int, help='Lower'
                                                               ' bound of the pixel percentile')
parser.add_argument('--percent_max', default=99, type=int, help='Upper'
                                                                ' bound of the pixel percentile')


def dcm2jpg(dcm_file, jpg_file, eh=False, percent_min=1, percent_max=99):
    logging.info('{}, Converting {}...'.format(
        time.strftime("%Y-%m-%d %H:%M:%S"), dcm_file))
    dcm = pydicom.dcmread(dcm_file, force=True)
    pixel_array = sitk.GetArrayFromImage(sitk.ReadImage(dcm_file))[0]

    if len(pixel_array.shape) > 2:
        pixel_array = pixel_array[..., 0]
    if hasattr(dcm, 'PhotometricInterpretation'):
        if dcm.PhotometricInterpretation == 'MONOCHROME1':
            pixel_array = cv2.bitwise_not(pixel_array)

    pixel_min, pixel_max = np.percentile(pixel_array,
                                         [percent_min, percent_max])

    image = np.clip(pixel_array, pixel_min, pixel_max)

    image = (image - pixel_min) * 255.0 / (pixel_max - pixel_min)

    if eh:
        image = cv2.equalizeHist(image.astype(np.uint8))
    else:
        image = image.astype(np.uint8)

    image_equalized = image.astype(np.uint8)
    assert cv2.imwrite(jpg_file, image_equalized), "Write failed"


def dcm_alloc(opts):
    dcm_file, args = opts
    file_name = dcm_file.split('/')[-1].strip('.dcm')
    jpg_file = os.path.join(args.jpg_path, file_name + '.jpg')
    print(jpg_file)
    try:
        dcm2jpg(dcm_file, jpg_file, args.equalize_hist,
                percent_min=args.percent_min,
                percent_max=args.percent_max)
    except Exception:
        traceback.print_exc()


def run(args):
    CPU_NUM = multiprocessing.cpu_count()
    assert args.num_workers <= CPU_NUM, \
        'num_workers:{} exceeds cpu_count:{}'.format(args.num_workers,
                                                     CPU_NUM)
    if not os.path.exists(args.jpg_path):
        os.mkdir(args.jpg_path)
    dcm_files = glob.glob(os.path.join(args.dcm_path, '*.dcm'))
    exist_jpg = glob.glob(os.path.join(args.jpg_path, '*.jpg'))
    filenames_exists = set(map(lambda x: x.split('/')[-1], exist_jpg))
    dcm_files_ = []
    for dcm_file in dcm_files:
        if dcm_file.split('/')[-1].replace('.dcm', '.jpg') not in \
                filenames_exists:
            dcm_files_.append(dcm_file)
    opts = [(dcm_file, args) for dcm_file in dcm_files_]
    pool = multiprocessing.Pool(processes=args.num_workers)
    pool.map(dcm_alloc, opts)


def main():
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()