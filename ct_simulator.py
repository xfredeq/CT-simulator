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

session = st.session_state
if 'run_id' not in session:
    session['run_id'] = 0


def on_param_change():
    session.run_id += 1


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

    alpha_step = st.slider('Delta alpha (step in degrees)', 0.1, 5.0, 4.0, 0.1, on_change=on_param_change())
    phi = st.slider("divergence (in degrees)", 0, 360, 180, 10)
    n = st.slider("number of detectors", 0, 400, 200, 10)

    st.subheader("Result generation options")
    option = st.radio("", const.GEN_OPTIONS, key=session.run_id)

    if option == "Clear":
        pass
    elif option == "Show intermediate steps":
        pass
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



