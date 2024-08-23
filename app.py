import streamlit as st
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from googletrans import Translator
from gtts import gTTS
import os

def gtts_text_to_speech(text, lang):
    try:
        tts = gTTS(text=text, lang=lang)
        audio_path = "output.mp3"
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        st.error(f"Error with gTTS: {e}")
        return None

def app_page():
    st.title("Image Captioning 📸")

    # Model loading and caching
    @st.cache_resource
    def load_models():
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        return model, feature_extractor, tokenizer

    # Initialize photo state in session
    if "photo" not in st.session_state:
        st.session_state["photo"] = "not done"

    # Navigation bar for file upload
    st.sidebar.title("Navigation")
    file_upload = st.sidebar.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    
    def change_photo_state():
        st.session_state["photo"] = "done"

    # Function to load image
    @st.cache_data
    def load_image(img):
        im = Image.open(img)
        return im

    # Layout for the main content
    c1, c2, c3 = st.columns([1, 2, 1])
    
    with c2:
        camera_photo = st.camera_input("Take a photo", on_change=change_photo_state)

    # Check if the caption generation checkbox is selected
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
            if file_upload:
                our_image = load_image(file_upload)
            elif camera_photo:
                our_image = load_image(camera_photo)
            else:
                our_image = None

            if our_image is not None:
                caption = predict_step(our_image)[0]  # Get the first caption from the list
                st.success(f"Generated Caption: {caption}")

                # Translation Section
                st.subheader("Translate the Caption🔠👀🤔🧐🌐🗨️")
                languages = {
                    "Assamese": "as", "Bengali": "bn", "Bodo": "brx", "Dogri": "doi", 
                    "Gujarati": "gu", "Hindi": "hi", "Kannada": "kn", "Kashmiri": "ks", 
                    "Konkani": "kok", "Maithili": "mai", "Malayalam": "ml", "English":"en",
                    "Manipuri (Meitei)": "mni", "Marathi": "mr", "Nepali": "ne", 
                    "Odia": "or", "Punjabi": "pa", "Sanskrit": "sa", "Santali": "sat", 
                    "Sindhi": "sd", "Tamil": "ta", "Telugu": "te", "Urdu": "ur"
                }

                target_language_name = st.selectbox('Select your target language', options=list(languages.keys()))
                target_language_code = languages[target_language_name]

                if st.button('Translate Caption'):
                    translator = Translator()
                    translation = translator.translate(caption, dest=target_language_code)
                    st.success(f"Translated Caption: {translation.text}")

                    # Text-to-Speech Section
                    st.subheader("Convert Caption to Speech 🎵")
                    audio_path = gtts_text_to_speech(translation.text, target_language_code)
                    if audio_path:
                        st.audio(audio_path, format='audio/mp3')
                    else:
                        st.error("Text-to-Speech conversion failed. Please try again.")
            else:
                st.error("Please upload an image or take a photo.")
        else:
            st.write("Photo not uploaded or captured.")
    elif st.checkbox("About"):
        st.subheader("About Image Captioning App")
        st.markdown("Built with Streamlit by [Abhi Varma](https://Abhi-varma-personal-website.streamlit.app/)")
        st.markdown("Demo application of the following model [credit](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning/)")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["App", "About"])

    if page == "App":
        app_page()
    elif page == "About":
        st.subheader("About Image Captioning App")
        st.markdown("Built with Streamlit by [Abhi Varma](https://Abhi-varma-personal-website.streamlit.app/)")
        st.markdown("Demo application of the following model [credit](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning/)")

if __name__ == "__main__":
    main()
