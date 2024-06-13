import streamlit as st
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
#pickle.load(open('energy_model.pkl', 'rb'))
#vocab = np.load('w2i.p', allow_pickle=True)
st.title("Image_Captioning_App")
@st.experimental_singleton
def load_models():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, feature_extractor, tokenizer
#st.text("Build with Streamlit and OpenCV")
if "photo" not in st.session_state:
	st.session_state["photo"]="not done"
c2, c3 = st.columns([2,1])
def change_photo_state():
	st.session_state["photo"]="done"
@st.cache
def load_image(img):
	im = Image.open(img)
	return im
uploaded_photo = c3.file_uploader("Upload Image",type=['jpg','png','jpeg'], on_change=change_photo_state)
camera_photo = c2.camera_input("Take a photo", on_change=change_photo_state)

#st.subheader("Detection")
if st.checkbox("Generate_Caption"):
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
   if st.session_state["photo"]=="done":
      if uploaded_photo:
         our_image= load_image(uploaded_photo)
      elif camera_photo:
         our_image= load_image(camera_photo)
      elif uploaded_photo==None and camera_photo==None:
          pass
         #our_image= load_image('image.jpg')
      st.success(predict_step(our_image))
elif st.checkbox("About"):
   st.subheader("About Image Captioning App")
   st.markdown("Built with Streamlit by [Soumen Sarker](https://soumen-sarker-personal-website.streamlit.app/)")
   st.markdown("Demo applicaton of the following model [credit](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning/)")