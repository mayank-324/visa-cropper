import streamlit as st
from PIL import Image
import os
import io

try:
    from visa_photo_processor import create_compliant_photo, DEFAULT_SPECIFICATIONS
except ImportError:
    st.error("Error: Could not import 'visa_photo_processor.py'. "
             "Make sure it's in the same directory as streamlit_app.py.")
    def create_compliant_photo(*args, **kwargs):
        return False, "Processing function not available."
    DEFAULT_SPECIFICATIONS = {"document_name": "Default (Unavailable)"}

ALL_SPECS = {
    "Generic Visa (Default)": DEFAULT_SPECIFICATIONS,
    "Canada Visa": {
        "document_name": "Canada Visa",
        "output_pixel_dims": {"width": 413, "height": 531}, # 35x45mm @ 300 DPI
        "background": {"type": "color", "color_rgb": (240, 240, 240), "allow_ai_background_removal": True}, # Light Gray
        "head_position": {"head_height_ratio": 0.68, "top_of_head_to_photo_top_ratio": 0.10}, # Approx 31-36mm head in 45mm height
        "auto_adjustments": {"lighting_method": "cv2_clahe"},
        "quality_checks": {"min_laplacian_variance": 80.0},
        "output_format": {"type": "JPEG", "quality": 95, "dpi": (300, 300)}
    },
    "Indian Passport/Visa (OCI Card may vary)": {
        "document_name": "Indian Passport/Visa",
        # India often asks for 2x2 inch (51x51mm)
        # 51mm / 25.4 * 300dpi = ~602 pixels
        "output_pixel_dims": {"width": 600, "height": 600}, # Approx 2x2 inch @ 300 DPI
        "background": {"type": "color", "color_rgb": (255, 255, 255), "allow_ai_background_removal": True}, # Plain White
        "head_position": {"head_height_ratio": 0.70, "top_of_head_to_photo_top_ratio": 0.12}, # Head should cover 60-75%
        "auto_adjustments": {"lighting_method": "cv2_clahe"},
        "quality_checks": {"min_laplacian_variance": 70.0},
        "output_format": {"type": "JPEG", "quality": 95, "dpi": (300, 300)}
    },
    "US Visa (Digital Upload)": {
        "document_name": "US Visa (Digital)",
        "output_pixel_dims": {"width": 600, "height": 600}, # Min 600x600, Max 1200x1200. Square.
        "background": {"type": "color", "color_rgb": (255, 255, 255), "allow_ai_background_removal": True}, # Plain White
        "head_position": {"head_height_ratio": 0.55, "top_of_head_to_photo_top_ratio": 0.20}, # Head 50-69% of image height. Eyes between 56-69% from bottom.
        "auto_adjustments": {"lighting_method": "cv2_clahe"},
        "quality_checks": {"min_laplacian_variance": 90.0},
        "output_format": {"type": "JPEG", "quality": 90, "dpi": (300, 300)} # DPI is less critical for digital, but good to set
    }
}

st.set_page_config(page_title="Visa Photo Generator", layout="centered")

st.title("📄 Visa & Passport Photo Generator")
st.markdown("""
Upload your photo, select the document type, and get a compliant photo in minutes!
**Disclaimer:** This tool attempts to meet official guidelines, but always double-check requirements
from official government sources. We are not responsible for rejected photos.
""")

uploaded_file = st.file_uploader("1. Upload Your Photo (Clear, frontal face recommended)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # displaying uploaded image
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Your Uploaded Photo", width=300)

        # creating temp directory to store
        temp_dir = "temp_uploads"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        temp_input_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.markdown("---")
        st.subheader("2. Select Document Type")
        doc_options = list(ALL_SPECS.keys())
        selected_doc_name = st.selectbox("Choose the document:", doc_options)

        selected_specifications = ALL_SPECS[selected_doc_name]

        # displaying select spec
        with st.expander("View Selected Document Specifications"):
            st.write(f"**Document:** {selected_specifications['document_name']}")
            st.write(f"**Output Size (pixels):** {selected_specifications['output_pixel_dims']['width']}w x {selected_specifications['output_pixel_dims']['height']}h")
            st.write(f"**Background Color (RGB):** {selected_specifications['background']['color_rgb']}")
            st.write(f"**AI Background Removal:** {'Enabled' if selected_specifications['background']['allow_ai_background_removal'] else 'Disabled'}")

        st.markdown("---")
        if st.button("🚀 Generate Compliant Photo", type="primary", use_container_width=True):
            st.subheader("3. Your Processed Photo")
            with st.spinner("Processing your photo... This might take a moment (especially with AI background removal)."):
                
                temp_output_filename = f"processed_{selected_doc_name.replace(' ', '_').lower()}_{uploaded_file.name}"
                temp_output_path = os.path.join(temp_dir, temp_output_filename)

                #calling function from processor script
                success, message = create_compliant_photo(
                    temp_input_path, 
                    temp_output_path, 
                    selected_specifications
                )

            if success:
                st.success(f"✅ Photo processed! {message.split('Issues:')[0].strip()}")
                if "Issues: None" not in message and "Issues:" in message:
                    st.warning(f"⚠️ Potential Issues: {message.split('Issues:')[1].strip()}")

                processed_image_display = Image.open(temp_output_path)
                st.image(processed_image_display, caption=f"Processed for: {selected_doc_name}", width=300)

                with open(temp_output_path, "rb") as file_to_download:
                    st.download_button(
                        label="⬇️ Download Processed Photo",
                        data=file_to_download,
                        file_name=f"compliant_{selected_doc_name.replace(' ', '_').lower()}_{uploaded_file.name}",
                        mime="image/jpeg" # Adjust if you allow PNG output
                    )

            else:
                st.error(f"❌ Processing Failed: {message}")
                st.info("Tips for better results: \n"
                        "- Use a clear photo with good, even lighting. \n"
                        "- Ensure your face is looking directly at the camera. \n"
                        "- Try a plain background if AI background removal struggles.")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.error("Please ensure the uploaded file is a valid image (JPG, PNG).")

else:
    st.info("Upload an image to get started.")