import streamlit as st
import numpy as np

from PIL import Image
from pydicom import dcmread

import functions as fun
import constants as const

st.set_page_config(
        page_title="CT Simulator",
        page_icon=":computer:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': const.ABOUT
        }
)


def set_default():
    st.session_state.alpha = 2.0
    st.session_state.phi = 180
    st.session_state.n = 200

    st.session_state.radio = "Clear"


def reset_radio():
    st.session_state.radio = "Clear"


st.markdown(const.AUTHORS, unsafe_allow_html=True)

title = st.title(const.TITLE)

st.sidebar.write(const.SIDEBAR_TEXT)

st.subheader("File upload")

file = None
image = None
patient_info = {}

dicom_checkbox = st.checkbox('use DICOM format')
if dicom_checkbox:
    file = st.file_uploader("Choose file", type=['dcm'])
    if file is not None:
        st.write("DICOM file uploaded succesfully")
        dc = dcmread(file)
        image = dc.pixel_array
        patient_info['id'] = dc.PatientID
        patient_info['name'] = str(dc.PatientName)
        st.write(patient_info)
else:
    file = st.file_uploader("Choose file", type=['jpg', 'jpeg', 'png'])
    if file is not None:
        image = Image.open(file)

if image is not None:
    image = np.array(image)

    st.subheader(f'Uploaded Image ({image.shape[0]}x{image.shape[1]})')
    st.image(image, caption="Uploaded image")

    cropped_image = fun.adjust_image(image)
    st.subheader(f'Adjusted grayscaled image ({cropped_image.shape[0]}x{cropped_image.shape[1]})')
    st.image(cropped_image, caption="Adjusted image")

    st.subheader("Define tomograph parameters")

    alpha_step = st.slider('Delta alpha (step in degrees)', 0.1, 5.0, 2.0, 0.1, key='alpha', on_change=reset_radio)
    phi = st.slider("divergence (in degrees)", 0, 360, 180, 10, key='phi', on_change=reset_radio)
    n = st.slider("number of detectors", 0, 400, 200, 10, key='n', on_change=reset_radio)
    default_button = st.button(label='Set default', on_click=set_default)

    st.subheader("Result generation options")
    option = st.radio("", const.GEN_OPTIONS, key='radio')

    if option == "Clear":
        st.write("Cleared")
    elif option == "Show intermediate steps":
        st.write("steps")
        sinogram = fun.make_sinogram(cropped_image, alpha_step, phi, n)

        st.subheader("Set number of iterations")
        iterations = st.slider("", 1, int(np.floor(360/alpha_step)), step=5)

        st.subheader("Sinogram")
        st.image(sinogram[0:iterations], caption=f'Sinogram ({iterations}x{sinogram.shape[1]})')

        filtered = fun.filter_sinogram(sinogram)

        st.subheader("Reconstructed image")
        reconstructed = fun.reconstruct_image(filtered, alpha_step, phi, n, cropped_image.shape[0], iterations)
        st.image(reconstructed, caption="Reconstructed image")

    else:
        st.subheader("Sinogram")
        sinogram = fun.make_sinogram(cropped_image, alpha_step, phi, n)
        st.image(sinogram, caption=f'Sinogram ({sinogram.shape[0]}x{sinogram.shape[1]})')

        st.subheader("Filtered sinogram")
        filtered = fun.filter_sinogram(sinogram)
        st.image(fun.normalize(filtered), caption="Filtered sinogram")

        st.subheader("Reconstructed image")
        reconstructed = fun.reconstruct_image(filtered, alpha_step, phi, n, cropped_image.shape[0])
        st.image(reconstructed, caption="Reconstructed image")
