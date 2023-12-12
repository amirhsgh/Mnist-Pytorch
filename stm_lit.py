import torchvision.transforms as transforms
import streamlit as st
from PIL import Image
import torch
from Model import Model
import time

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1303,), (0.3081,))
])


model = Model()
model_state = torch.load("./mnist_cnn.pt")
model.load_state_dict(model_state)
model.eval()

st.title("Predict a number by image")

st.text("""
    You can upload a Image number in 28 * 28 pixel and got the number result
""")

image = st.file_uploader(label="Image uploader")
if image:
    if image.name.split('.')[1] in ['png', 'jpeg', 'jpg']:
        st.image("./" + image.name, width=100)
        st.toast(":green[Image uploaded successfully]")
        with st.spinner('Wait for our model . . .'):
            time.sleep(4)
            st.toast(':green[Done!]')
            test_image = Image.open("./" + image.name).convert("L")
            test_data = transform(test_image)

            with torch.no_grad():
                output = model(test_data.unsqueeze(0))

            result = output.argmax(dim=1, keepdim=True)
            if str(result.item()) == image.name.split('.')[0].split('_')[1]:
                st.subheader("The result of this image is :green[{}]".format(result.item()))
                st.toast(":green[Yoooo our predication is correct!!]")
            else:
                st.subheader("The result of this image is :green[{}] and your model answer :red[{}]".format(image.name.split('.')[0].split('_')[1], result.item()))
                st.toast(":red[your prediction not correct!!]")
    else:
        st.toast(":red[wrong type please send png,jpeg or jpg]")