import streamlit as st
from PIL import Image
import io
import time
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

# Initialize session state for user data and login status
if 'users' not in st.session_state:
    st.session_state['users'] = {}
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'current_user' not in st.session_state:
    st.session_state['current_user'] = None
if 'forgot_password' not in st.session_state:
    st.session_state['forgot_password'] = False

def login(username, password):
    users = st.session_state['users']
    if username in users and users[username]['password'] == password:
        st.session_state['logged_in'] = True
        st.session_state['current_user'] = username
        st.success("Logged in successfully!")
    else:
        st.error("Invalid username or password")

def logout():
    st.session_state['logged_in'] = False
    st.session_state['current_user'] = None
    st.success("Logged out successfully!")

def register_user(first_name, last_name, email, sex, password, photos):
    users = st.session_state['users']
    if email in users:
        st.error("User already registered with this email!")
    else:
        photo_list = []
        for photo in photos:
            image = Image.open(photo)
            byte_array = io.BytesIO()
            image.save(byte_array, format='PNG')
            photo_list.append(byte_array.getvalue())
        
        users[email] = {
            'first_name': first_name,
            'last_name': last_name,
            'sex': sex,
            'password': password,
            'photos': photo_list
        }
        st.success("Registration successful! Please log in.")

def forgot_password_page():
    st.title("Forgot Password")
    email = st.text_input("Enter your email")
    if st.button("Reset Password"):
        users = st.session_state['users']
        if email in users:
            st.success(f"Password reset instructions have been sent to {email}.")
        else:
            st.error("This email is not registered.")

def login_page():
    st.title("Login")
    username = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        login(username, password)
    if st.button("Forgot Password"):
        st.session_state['forgot_password'] = True

def register_page():
    st.title("Register")
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    email = st.text_input("Email")
    sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    photos = st.file_uploader("Upload Photos (min 5)", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
    
    if st.button("Register"):
        if password == confirm_password:
            if len(photos) < 5:
                st.error("You must upload at least 5 photos.")
            else:
                register_user(first_name, last_name, email, sex, password, photos)
        else:
            st.error("Passwords do not match!")

def app_page():
    st.title("Image Captioning App")

    @st.cache_resource
    def load_models():
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        return model, feature_extractor, tokenizer

    if "photo" not in st.session_state:
        st.session_state["photo"] = "not done"

    c2, c3 = st.columns([2, 1])

    def change_photo_state():
        st.session_state["photo"] = "done"

    @st.cache_data
    def load_image(img):
        im = Image.open(img)
        return im

    uploaded_photo = c3.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'], on_change=change_photo_state)
    camera_photo = c2.camera_input("Take a photo", on_change=change_photo_state)

    if st.checkbox("Generate Caption"):
        model, feature_extractor, tokenizer = load_models()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        max_length = 16
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

        def predict_step(our_image):
            if our_image.mode != "RGB":
                our_image = our_image.convert(mode="RGB")
            pixel_values = feature_extractor(images=our_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            output_ids = model.generate(pixel_values, **gen_kwargs)
            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]
            return preds

        if st.session_state["photo"] == "done":
            if uploaded_photo:
                our_image = load_image(uploaded_photo)
            elif camera_photo:
                our_image = load_image(camera_photo)
            else:
                our_image = None

            if our_image is not None:
                caption = predict_step(our_image)
                st.success(caption)
    elif st.checkbox("About"):
        st.subheader("About Image Captioning App")
        st.markdown("Built with Streamlit by [Abhi Varma](https://Abhi-varma-personal-website.streamlit.app/)")
        st.markdown("Demo application of the following model [credit](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning/)")

def main():
    st.sidebar.title("Navigation")
    if st.session_state['logged_in']:
        page = st.sidebar.radio("Go to", ["App", "Logout"])
        if page == "Logout":
            logout()
            st.experimental_rerun()
    else:
        page = st.sidebar.radio("Go to", ["Login", "Register"])

    if st.session_state['logged_in']:
        if page == "App":
            app_page()
        else:
            st.write(f"Welcome, {st.session_state['users'][st.session_state['current_user']]['first_name']}!")
    else:
        if st.session_state['forgot_password']:
            forgot_password_page()
        else:
            if page == "Login":
                login_page()
            elif page == "Register":
                register_page()

if __name__ == "__main__":
    main()
