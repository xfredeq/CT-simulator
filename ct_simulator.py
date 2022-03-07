import streamlit as st
from PIL import Image
from pydicom import dcmread

st.set_page_config(
        page_title="CT Simulator",
        page_icon=":computer:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help'    : 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About'       : "# This is a header. This is an *extremely* cool app!"
        }
)

st.markdown("<h3 style='text-align: left; color: grey;'>Authors:</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: left; color: grey;'>Mateusz Frąckowiak 145264</h4>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: left; color: grey;'>Kamil Niżnik 145XXX</h4>", unsafe_allow_html=True)

title = st.title('Computer Tomograph Simulation')

st.sidebar.write("text")

st.subheader("File upload")

file = None
image = None
patient_info = {}

if st.checkbox('use DICOM format'):
    file = st.file_uploader("Choose file", type=['dcm'])
    if file is not None:
        st.write("DICOM file uploaded succesfully")
        dc = dcmread(file)
        image = dc.pixel_array
else:
    file = st.file_uploader("Choose file", type=['jpg', 'jpeg', 'png'])
    if file is not None:
        image = Image.open(file)

if image is not None:
    st.image(image, caption="Uploaded image")
