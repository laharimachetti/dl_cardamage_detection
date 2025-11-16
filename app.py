import streamlit as st
from model_helper import predict

# ----- GLOBAL PAGE STYLE -----
st.markdown("""
    <style>
        body {
            background-color: #f4f7fb;
        }
        .upload-box {
            background: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div style="width:100%; text-align:center;">
        <h1 style="color:#1f2937;">Vehicle Damage Detection</h1>
    </div>
    """,
    unsafe_allow_html=True
)


# ----- LEFT ART + RIGHT UPLOADER -----
col1, col2 = st.columns([1, 2])

with col1:
    st.image("vehicle.png", use_column_width=True)

with col2:
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload the file", type=["jpg", "png"])

    if uploaded_file:
        image_path = "temp_file.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        prediction = predict(image_path)
        st.info(f"Predicted Class: {prediction}")

    st.markdown("</div>", unsafe_allow_html=True)

# ----- IMAGE BELOW -----
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded File", use_container_width=True)
