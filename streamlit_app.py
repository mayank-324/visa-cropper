import streamlit as st
from PIL import Image
import os
import io
import sys

st.set_page_config(page_title="Visa Photo Generator", layout="centered")

# st.write(f"DEBUG: Current working directory: {os.getcwd()}")
# st.write(f"DEBUG: Python sys.path: {sys.path}")
# st.write(f"DEBUG: Files in current directory: {os.listdir('.')}")
# if os.path.exists("visa_photo_processor.py"):
#     st.write("DEBUG: visa_photo_processor.py FOUND in current directory.")
# else:
#     st.write("DEBUG: visa_photo_processor.py NOT FOUND in current directory.")

try:
    from visa_photo_processor import create_compliant_photo, DEFAULT_SPECIFICATIONS
    # st.write("DEBUG: Successfully imported from visa_photo_processor.py") # Add this
    # st.write(f"DEBUG: Imported DEFAULT_SPECIFICATIONS: {DEFAULT_SPECIFICATIONS.get('document_name', 'Name missing')}")
except ImportError as e:
    st.error(f"Error: Could not import 'visa_photo_processor.py'. Exception: {e} "
             "Make sure it's in the same directory as streamlit_app.py.")
    def create_compliant_photo(*args, **kwargs):
        return False, "Processing function not available."
    DEFAULT_SPECIFICATIONS = {
        "document_name": "Default (Unavailable - Import Failed)",
        "output_pixel_dims": {"width": 0, "height": 0}, # Add dummy
        "background": {"type": "color", "color_rgb": (0,0,0), "allow_ai_background_removal": False}, # Add dummy
        "head_position": {"head_height_ratio": 0, "top_of_head_to_photo_top_ratio": 0}, # Add dummy
        "auto_adjustments": {"lighting_method": "none"}, # Add dummy
        "quality_checks": {"min_laplacian_variance": 0}, # Add dummy
        "output_format": {"type": "JPEG", "quality": 0, "dpi": (0,0)} # Add dummy
    }

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
    },
    "Russian Federation Visa": {
        "document_name": "Russian Federation Visa",
        # Standard size is 35mm x 45mm. [5, 6, 17, 20, 21, 23, 24]
        # Resolution: 600 DPI is often mentioned for good quality prints. [23, 24]
        # 35mm @ 600 DPI = (35 / 25.4) * 600 = ~827 pixels
        # 45mm @ 600 DPI = (45 / 25.4) * 600 = ~1063 pixels
        "output_pixel_dims": {"width": 827, "height": 1063},
        "background": {"type": "color", "color_rgb": (255, 255, 255), "allow_ai_background_removal": True}, # Plain white or light-colored. [5, 6, 17, 20, 21] White is safest.
        "head_position": {
            # Face should cover about 50% of the photo area. [5, 6]
            # Some sources say head height 32-36mm or around 33mm for a 45mm photo height. [24, 22] This is ~71-80%. Let's aim for ~75%.
            # Or 70-80% of image. [21]
            # Top of head to top of photo: ~5mm. [23, 24] (5mm / 45mm = ~0.11)
            "head_height_ratio": 0.73, # (Approx (33mm / 45mm))
            "top_of_head_to_photo_top_ratio": 0.11 # (Approx (5mm / 45mm))
        },
        "auto_adjustments": {"lighting_method": "cv2_clahe"},
        "quality_checks": {"min_laplacian_variance": 80.0},
        "output_format": {"type": "JPEG", "quality": 95, "dpi": (600, 600)} # High quality print often implies 600 DPI.
    },

    "Japan Visa": {
        "document_name": "Japan Visa",
        # Commonly 45mm x 45mm. [2, 3, 10, 15, 16, 19, 26] Some mention 35x45mm also acceptable. [10] Let's go with 45x45mm as primary.
        # Resolution: 300 DPI [2] or 600 DPI. [3, 15, 19, 26] Let's use 600 for better quality if possible.
        # 45mm @ 600 DPI = (45 / 25.4) * 600 = ~1063 pixels
        "output_pixel_dims": {"width": 1063, "height": 1063},
        "background": {"type": "color", "color_rgb": (255, 255, 255), "allow_ai_background_removal": True}, # Plain white. [3, 16, 19, 26] Some sources say white or light gray [2, 10, 15]. White is safest.
        "head_position": {
            # Head height: ~27mm. [2, 3]
            # Top of photo to top of hair: ~7.5mm. [2, 3]
            # For a 45mm photo height:
            # head_height_ratio = 27/45 = 0.60
            # top_of_head_to_photo_top_ratio = 7.5/45 = ~0.167
            # Some sources say face 70-80% of photo. [15, 16] 27mm in 45mm is 60%. This can vary by interpretation (face vs full head). Let's use the mm values.
            "head_height_ratio": 0.60,
            "top_of_head_to_photo_top_ratio": 0.16
        },
        "auto_adjustments": {"lighting_method": "cv2_clahe"},
        "quality_checks": {"min_laplacian_variance": 80.0},
        "output_format": {"type": "JPEG", "quality": 95, "dpi": (600, 600)}
    },

    "China Visa (Paper & Digital General)": {
        "document_name": "China Visa",
        # Paper Photo: 33mm width x 48mm height. [4, 7, 8, 9, 12, 14]
        # Digital Photo: 354-420 pixels (width) x 472-560 pixels (height). [7, 9, 11, 14]
        # Let's target the mid-range for digital or calculate from paper at a good DPI.
        # 33mm @ 300 DPI = (33 / 25.4) * 300 = ~390 pixels
        # 48mm @ 300 DPI = (48 / 25.4) * 300 = ~567 pixels (This is slightly outside the higher digital spec height, but often digital specs have ranges)
        # Let's use the common digital target: Width: 390px (approx 33mm @ 300dpi), Height: 530px (mid-range of 472-560)
        "output_pixel_dims": {"width": 390, "height": 530}, # Targeting common digital size.
        "background": {"type": "color", "color_rgb": (255, 255, 255), "allow_ai_background_removal": True}, # Plain white or close to white. [4, 7, 8, 9, 12, 13, 14]
        "head_position": {
            # Head height (chin to crown): 28mm-33mm. [4, 7, 8, 9, 12] Let's aim for ~30.5mm.
            # Head width: 15mm-22mm. [4, 7, 9, 12] (Not directly used for ratio, but good to know)
            # Space from upper edge of image to crown of head: 3mm-5mm. [7, 9] Let's aim for ~4mm.
            # For a 48mm photo height:
            # head_height_ratio = 30.5 / 48 = ~0.635
            # top_of_head_to_photo_top_ratio = 4 / 48 = ~0.083
            # For a 530px photo height (our digital target):
            # If physical top margin is 4mm in a 48mm photo, then in a 530px photo, the margin is (4/48)*530 = ~44px. Ratio: 44/530 = ~0.083
            # If physical head height is 30.5mm in a 48mm photo, then in a 530px photo, head height is (30.5/48)*530 = ~337px. Ratio: 337/530 = ~0.635
            "head_height_ratio": 0.635,
            "top_of_head_to_photo_top_ratio": 0.083
        },
        "auto_adjustments": {"lighting_method": "cv2_clahe"},
        "quality_checks": {"min_laplacian_variance": 70.0},
        "output_format": {"type": "JPEG", "quality": 90, "dpi": (300, 300)}, # Digital photos often have file size limits (40KB-120KB JPEG [7, 8, 9, 11])
        "file_size_kb_limits": {"min": 40, "max": 120} # Add this custom key if you implement file size check/compression
    }
}

st.title("📄 Visa & Passport Photo Generator")
st.markdown("""
Upload your photo, select the document type, and get a compliant photo in minutes!
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