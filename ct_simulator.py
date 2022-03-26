from datetime import datetime
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image
from pydicom import dcmread

import constants as const
import functions as fun

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


title = st.title(const.TITLE)

st.sidebar.markdown(const.AUTHORS, unsafe_allow_html=True)
st.sidebar.write(const.SIDEBAR_TEXT)

filter_checkbox = st.sidebar.checkbox('Filter Sinogram', value=True, on_change=reset_radio)

dicom_save_checkbox = st.sidebar.checkbox("Save in Dicom Format")

st.subheader("File upload")

file = None
image = None
patient_info = {}

dicom_checkbox = st.checkbox('use DICOM format')
if dicom_checkbox:
    file = st.file_uploader("Choose file", type=['dcm'])
    if file is not None:
        st.info("DICOM file uploaded succesfully")
        dc = dcmread(file)
        image = dc.pixel_array

        patient_info = fun.get_patient_info(dc)
        st.write(dc.__dict__)
        st.write(patient_info)
else:
    file = st.file_uploader("Choose file", type=['jpg', 'jpeg', 'png'])
    if file is not None:
        image = Image.open(file)

if image is not None:
    image = np.array(image)
    cropped_image = fun.adjust_image(image)

    input1, input2 = st.columns(2)
    with input1:
        st.subheader(f'Uploaded Image ({image.shape[0]}x{image.shape[1]})')
        st.image(image, caption="Uploaded image")

    with input2:
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
    else:
        if option == "Show intermediate steps":

            sinogram = fun.make_sinogram(cropped_image, alpha_step, phi, n)

            st.subheader("Sinogram generation steps")
            sinogram_step = st.slider("", 1, int(np.floor(360 / alpha_step)), step=5)

            st.subheader("Sinogram")
            st.image(sinogram[0:sinogram_step], caption=f'Sinogram ({sinogram_step}x{sinogram.shape[1]})')

            st.subheader("Tomograph rotation degree")
            iterations = st.slider("", 1, 360, step=5)

            if filter_checkbox:
                filtered = fun.filter_sinogram(sinogram)
                st.subheader("Reconstructed image")
                reconstructed = fun.reconstruct_image(filtered, alpha_step, phi, n, cropped_image.shape[0], iterations)
                st.image(reconstructed, caption="Reconstructed image")
            else:
                st.subheader("Reconstructed image")
                reconstructed = fun.reconstruct_image(sinogram, alpha_step, phi, n, cropped_image.shape[0], iterations)
                st.image(reconstructed, caption="Reconstructed image")
        else:
            if filter_checkbox:
                sin1, sin2, rec = st.columns(3)
            else:
                sin1, rec = st.columns(2)

            with sin1:
                st.subheader("Sinogram")
                sinogram = fun.make_sinogram(cropped_image, alpha_step, phi, n)
                st.image(sinogram, caption=f'Sinogram ({sinogram.shape[0]}x{sinogram.shape[1]})')

            if filter_checkbox:
                with sin2:
                    st.subheader("Filtered sinogram")
                    filtered = fun.filter_sinogram(sinogram)
                    st.image(fun.normalize(filtered), caption="Filtered sinogram")

                reconstructed = fun.reconstruct_image(filtered, alpha_step, phi, n, cropped_image.shape[0])
            else:
                reconstructed = fun.reconstruct_image(sinogram, alpha_step, phi, n, cropped_image.shape[0])

            with rec:
                st.subheader("Reconstructed image")
                st.image(reconstructed, caption="Reconstructed image")

        st.subheader('Save file')

        with st.form("File Name"):
            file_name = st.text_input("File name", value=file.name, placeholder="file_name.jpg")
            if dicom_save_checkbox:
                st.write("Dicom metadata:")

                form_col1, form_col2 = st.columns(2)
                with form_col1:
                    patient_info['id'] = str(st.number_input("PatientID", min_value=0, value=int(patient_info['id'])))
                    date = st.date_input("Date", value=datetime.now())
                with form_col2:
                    patient_info['name'] = st.text_input("PatientName", value=patient_info['name'])
                    time = st.time_input("Time", value=datetime.now())

                patient_info['comments'] = st.text_input("ImageComments", value=patient_info['comments'])
                patient_info['date'] = datetime.combine(date, time).timestamp()

            confirm_filename = st.form_submit_button("Confirm")

        image_data = BytesIO()
        save_img = Image.fromarray(np.uint8(reconstructed * 255), 'L')
        save_img.save(image_data, format="JPEG")

        file_name = fun.adjust_filename(file_name, dicom_save_checkbox)

        if dicom_save_checkbox:
            "save in dicom format"
            dicom_file = fun.create_dicom(save_img, patient_info)
            st.download_button("Download dicom file", data=dicom_file, file_name=file_name)
        else:
            st.download_button("Download reconstructed image", data=image_data, file_name=file_name)


else:
    st.warning("File has not been uploaded")
