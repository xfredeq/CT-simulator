import math
import re
from io import BytesIO

import numpy as np
import pydicom._storage_sopclass_uids
import skimage.color
import skimage.draw
import streamlit as st
from PIL.Image import Image
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
from skimage.exposure import rescale_intensity
from skimage.util import img_as_ubyte


def _convert_coords(coords, r):
    return coords[0] + r, -coords[1] + r


def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def adjust_image(image):
    if len(image.shape) > 2:
        image = skimage.color.rgb2gray(image)

    if np.max(image) > 1:
        image = normalize(image)

    y, x = image.shape
    if y < x:
        fill = np.zeros(((x - y) // 2, x))
        image = np.vstack((fill, image, fill))
    elif x < y:
        fill = np.zeros((y, (y - x) // 2))
        image = np.hstack((fill, image, fill))

    dimension = min(image.shape)
    image = image[0:dimension, 0:dimension]

    return image


@st.experimental_memo
def make_sinogram(image, alpha_step=4, phi=180, n=200):
    alpha_step = np.deg2rad(alpha_step)
    phi = np.deg2rad(phi)

    img_size = image.shape[0]
    r = 0.5 * img_size

    iterations = math.ceil(2 * np.pi / alpha_step)
    sinogram = np.zeros((iterations, n))

    alpha = 0
    for iteration in range(iterations):
        x_e = r * np.cos(alpha)
        y_e = r * np.sin(alpha)
        x_e, y_e = _convert_coords((x_e, y_e), r)

        for i in range(n):
            x_d = r * np.cos(alpha + np.pi - phi / 2 + i * (phi / (n - 1)))
            y_d = r * np.sin(alpha + np.pi - phi / 2 + i * (phi / (n - 1)))
            x_d, y_d = _convert_coords((x_d, y_d), r)

            coords = skimage.draw.line_nd([x_e, y_e], [x_d, y_d])

            coords[0][coords[0] >= img_size] -= 1
            coords[1][coords[1] >= img_size] -= 1

            points = image[coords[1], coords[0]]

            sinogram[iteration, i] = np.mean(points)

        alpha += alpha_step

    sinogram = normalize(sinogram)
    return sinogram


def _make_kernel(size):
    half_size = math.floor(size / 2)
    kernel = np.zeros(half_size)
    center = math.floor(half_size / 2)

    for i in range(0, half_size):
        k = i - center
        if k % 2 != 0:
            kernel[i] = (-4 / (np.pi * np.pi)) / (k * k)

    kernel[center] = 1
    return kernel


def filter_sinogram(sinogram):
    kernel = _make_kernel(sinogram.shape[1])

    for i in range(len(sinogram)):
        sinogram[i, :] = np.convolve(sinogram[i, :], kernel, mode="same")

    return sinogram


@st.experimental_memo
def reconstruct_image(sinogram, alpha_step, phi, n, img_size, iterations=0):
    alpha_step = np.deg2rad(alpha_step)
    phi = np.deg2rad(phi)

    image = np.zeros((img_size, img_size))
    r = 0.5 * img_size

    iterations = math.ceil(2 * np.pi * iterations / (alpha_step * 360)) if iterations != 0 else math.ceil(
            2 * np.pi / alpha_step)

    alpha = 0
    for iteration in range(iterations):
        x_e = r * np.cos(alpha)
        y_e = r * np.sin(alpha)
        x_e, y_e = _convert_coords((x_e, y_e), r)

        for i in range(n):
            x_d = r * np.cos(alpha + np.pi - phi / 2 + i * (phi / (n - 1)))
            y_d = r * np.sin(alpha + np.pi - phi / 2 + i * (phi / (n - 1)))
            x_d, y_d = _convert_coords((x_d, y_d), r)

            coords = skimage.draw.line_nd([x_e, y_e], [x_d, y_d])

            coords[0][coords[0] >= img_size] -= 1
            coords[1][coords[1] >= img_size] -= 1

            image[coords[1], coords[0]] += sinogram[iteration, i]

        alpha += alpha_step

    image = normalize(image)

    return image


def get_patient_info(dc):
    info = {}
    try:
        info['id'] = int(dc.PatientID)
    except (ValueError, AttributeError):
        info['id'] = -1

    try:
        info['name'] = str(dc.PatientName)
    except AttributeError:
        info['name'] = ""

    try:
        info['sex'] = dc.PatientSex
    except AttributeError:
        info['sex'] = None

    try:
        info['weight'] = dc.PatientWeight
    except AttributeError:
        info['weight'] = -1

    try:
        info['comments'] = str(dc.ImageComments)
    except AttributeError:
        info['name'] = ""

    try:
        info['date'] = dc.ContentDate
    except AttributeError:
        info['date'] = None

    try:
        info['time'] = dc.ContentTime
    except AttributeError:
        info['time'] = None

    return info


def adjust_filename(file_name: str, dicom_save: bool):
    file_name = re.sub(r'\s+', '', file_name)
    file_name = file_name.split('.')[0]
    if dicom_save:
        file_name += '.dcm'
    else:
        file_name += '.jpg'
    return file_name


def convert_image_to_ubyte(img):
    return img_as_ubyte(rescale_intensity(img, out_range=(0.0, 1.0)))


def create_dicom(img: Image, patient_data: dict):
    # Populate required values for file meta information
    meta = Dataset()
    meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(None, {}, preamble=b"\0" * 128)
    ds.file_meta = meta

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID

    ds.PatientName = patient_data["name"]
    ds.PatientID = patient_data["id"]
    ds.ImageComments = patient_data["comments"]
    ds.PatientSex = patient_data["sex"]
    ds.PatientWeight = patient_data["weight"]
    ds.ContentDate = str(patient_data['date'])
    ds.ContentTime = str(patient_data['time'])


    ds.Modality = "CT"
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.FrameOfReferenceUID = pydicom.uid.generate_uid()

    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.SamplesPerPixel = 1
    ds.HighBit = 7

    ds.ImagesInAcquisition = 1
    ds.InstanceNumber = 1

    ds.Rows, ds.Columns = img.size

    ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0

    pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)

    ds.PixelData = img.tobytes()

    dicom_file = BytesIO()
    ds.save_as(dicom_file, write_like_original=False)
    return dicom_file


def calculate_rmse(img1, img2):
    if(img1.shape != img2.shape):
        raise Exception("[ERROR] calculate_rmse(): img1.shape != img2.shape")

    if(np.min(img1) < 0 or np.max(img1) > 1):
        raise Exception("[ERROR] calculate_rmse(): img1 not normalized")

    if(np.min(img2) < 0 or np.max(img2) > 1):
        raise Exception("[ERROR] calculate_rmse(): img2 not normalized")

    N = img1.shape[0] * img1.shape[1]

    return np.sqrt(np.sum((img1 - img2)**2) / N)  