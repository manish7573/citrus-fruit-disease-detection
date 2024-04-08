import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu
from englisttohindi.englisttohindi import EngtoHindi

st.set_page_config(page_title='Detect!t',page_icon="./letter-d.png")

def model_prediction(test_image):
    model = tf.keras.models.load_model("testing2.h5",compile=False)
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

def preventive_measures(disease):
    measures = ""
    if disease == 'Greening-fruit':
        measures = ("1. Use reflective mulch to deter insect vectors.\n"
                    "2. Apply systemic insecticides to control psyllid populations.\n"
                    "3. Remove and destroy infected trees.\n"
                    "4. Regularly monitor trees for signs of infection.\n")
    elif disease == 'greening-leaf':
        measures = ("1. Choose resistant varieties.\n"
                    "2. Remove infected plant material.\n"
                    "3. Apply copper sprays preventively.\n"
                    "4. Implement quarantine measures.\n"
                    "5. Monitor and detect symptoms early.")
    elif disease == 'Scab-fruit':
        measures = ("1. Remove and destroy infected leaves and fruit.\n"
                    "2. Prune trees for better airflow.\n"
                    "3. Apply fungicides during high disease pressure.\n"
                    "4. Use drip irrigation to minimize leaf wetness.\n"
                    "5. Consider planting scab-resistant varieties.")
    elif disease == 'Melanose-leaf':
        measures = ("1. Choose resistant varieties.\n"
                    "2. Remove infected plant material.\n"
                    "3. Apply copper sprays preventively.\n"
                    "4. Implement quarantine measures.\n"
                    "5. Monitor and detect symptoms early.")
    elif disease == 'Black spot-leaf':
        measures = ("1. Prune infected branches regularly.\n"
                    "2. Keep foliage dry by watering at the base..\n"
                    "3. copper-based fungicides.\n"
                    "4. Remove fallen leaves and debris from around trees.\n")
    elif disease == 'black spot-fruit':
        measures = ("1. Remove infected plant material.\n"
                    "2. Apply copper sprays during outbreaks.\n"
                    "3. Prune infected branches below symptoms.\n"
                    "4. Minimize overhead irrigation.\n"
                    "5. Implement strict quarantine measures.")
    elif disease == 'citrus-canker-fruit':
        measures = ("1. Choose resistant varieties.\n"
                    "2. Remove infected plant material.\n"
                    "3. Apply copper sprays preventively.\n"
                    "4. Implement quarantine measures.\n"
                    "5. Monitor and detect symptoms early.")
    elif disease == 'canker-leaf':
        measures = ("1. Use reflective mulch to deter insect vectors.\n"
                    "2. Apply systemic insecticides to control psyllid populations.\n"
                    "3. Remove and destroy infected trees.\n"
                    "4. Regularly monitor trees for signs of infection.\n")
    else:
        measures = "No specific preventive measures found for the predicted disease."

    return measures

# 1. as sidebar menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'About', 'Features','Disease Recognition'], 
        icons=['house', 'book','clipboard-data','search'], menu_icon="cast", default_index=0)
    selected

#Main pagerub
if (selected=="Home"): 
    st.header("Citrus Disease Detection Using Machine Learning")
    st.image("./home.jpg")
    st.markdown("""
    Welcome to the Citrus Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying citrus diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About
elif(selected=="About"):
    st.header("About")
    st.subheader("Introduction:")
    paragraph = """Citrus fruit detection plays a vital role in various industries, including agriculture and food processing. Accurately identifying and classifying citrus fruits such as oranges, lemons, and limes is crucial for quality control, sorting, and grading processes. Traditional methods of fruit inspection are time-consuming and labor-intensive, highlighting the need for automated detection systems.

In recent years, advancements in computer vision and machine learning have revolutionized fruit detection, enabling fast and accurate identification of citrus fruits based on their visual characteristics. These technologies have the potential to streamline fruit sorting processes, reduce waste, and improve overall efficiency in the citrus industry.

This project aims to develop a citrus fruit detection system using deep learning techniques. By leveraging the power of convolutional neural networks (CNNs) and image processing algorithms, we can create a robust solution capable of accurately detecting and classifying citrus fruits in real-time."""
    st.write(paragraph)
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this kaggle repo.
                This dataset consists of about 12K rgb images of healthy and diseased crop fruit and leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                Citrus Leaf Disease Image ( Orange Leaf)
                Types of images
                1. Black spot Leaf
                2. Canker Leaf
                3. Greening Leaf
                4. Healthy Leaf
                5. Melanose Leaf
                6. Scab Fruit
                7. Black Sot Fruit
                8. Greening Fruit
                9. Citrus Canker Fruit
                10. Healthy Fruit

                Citrus fruit [click here](https://www.kaggle.com/datasets/jonathansilva2020/dataset-for-classification-of-citrus-diseases)

                Citrus Leaf [click here](https://www.kaggle.com/datasets/myprojectdictionary/citrus-leaf-disease-image)
                """)
    


#Features
elif(selected=="Features"):
    st.header("Features")
    st.subheader("Features about the Project 📝🔍")
    st.markdown("""             
    ### Just need to click and upload fruit image: 
    This feature implies a user-friendly interface where users can simply click a button to upload an image of a citrus fruit for disease detection. This ease of use can encourage more users to utilize the system.
                
    ### Provides the cause and solution of the identified diseases: 
    After identifying a disease in a citrus fruit, the system can provide information on the cause of the disease and possible solutions or treatments. This can help farmers take appropriate action to manage the disease and protect their crops.
                    
    ### Supports around different citrus fruit:
    This feature indicates that the system is capable of detecting diseases in various types of citrus fruits. This versatility makes the system useful for a wide range of citrus fruit producers, regardless of the specific type of citrus fruit they are growing.
    """)
    

#Prediction
elif(selected=="Disease Recognition"):
    st.header("Disease Recognition")
    st.subheader("Test Your Fruit:")

    option = st.selectbox('Choose an input Image option:',
                          ('--select option--','Upload', 'Camera'))
    
    if(option=="Upload"):
        test_image=st.file_uploader("Choose an Image:")
        if(st.button("Show Image")):
          st.image(test_image,width=4,use_column_width=True)

    elif(option=="Camera"):
        test_image=st.camera_input("Capture an Image:")
        st.image(test_image,width=4,use_column_width=True)

    predicted_class = None
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        class_name=['Black spot-leaf',
                    'Greening-fruit',
                    'Melanose-leaf',
                    'Scab-fruit',
                    'black-spot-fruit',
                    'canker-leaf',
                    'citrus-canker-fruit',
                    'greening-leaf',
                    'healthy-fruit',
                    'healthy-leaf']
        predicted_class = class_name[result_index]
        if(predicted_class=="healthy-fruit"):
            st.success(f"Model is predicting it's a {predicted_class}")
        else:
            st.error(f"Model is predicting it's a {predicted_class}")
        
        co1, co2 = st.columns(2)
        with co1:
            st.write("Prevention Measures:")
            prevention_measures = preventive_measures(predicted_class)
            st.info(prevention_measures)
        

        with co2:
            res1 = EngtoHindi("Prevention Measures:")
            translated_measures = []
            for line in prevention_measures.strip().split('\n'):
                res = EngtoHindi(line.strip())
                translated_measures.append(res.convert)
            
            my_string = '\n'.join(translated_measures)

            # Display the translated header and measures
            st.write(res1.convert)
            st.info(my_string)
