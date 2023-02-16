import streamlit as st
import pandas as pd
import joblib
import altair as alt
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# -------------------------------------------------------------------------MODELS--------------------------------------------------------------------#
clf_svm = joblib.load("../Persisted Models/svm-model.joblib")
clf_multinomial_nb = joblib.load("../Persisted Models/multinomial-model.joblib")
clf_gradient = joblib.load("../Persisted Models/gradient-model.joblib")
clf_knn = joblib.load("../Persisted Models/knn-model.joblib")

processed_text_df = pd.read_csv("../Data/processed_text.csv")
# -------------------------------------------------------------------------MAIN()--------------------------------------------------------------------#
def main():
    st.title("Emotions NLP Text Classification")
    menu = ["Home","Predict Emotions","Data Monitor"]
    choice = ["SVM","KNearestNeighborsClassifier","GradientBoostingClassifier",
             "MultinomialNB"]

    choices = st.sidebar.selectbox("Menu",menu)
# ----------------------------------------------------------------HOME-----------------------------------------------------------------------------#
    if choices == "Home":
        st.subheader("Home-Emotions Text Classification")
        col7,col8 = st.columns(2)
        with col7:
            st.write("Hello👋🏽,Thank you for visiting my NLP project model deployment programe,this nlp model is related to emotions,for eg: you recieved a msg from one of your close friend but you couldn't tell wheater he is tying that in a frightened mood or a sad mood,if you are just a friend you would leave the situation to wash out of your brain but if you are a close friend to him i am sure that you will be curious to ask him, but there is the problem if he is in a sad mood he wouldn't like if you ask him what mood he is,at that time you could use this model to predict in which mood he is.")
        
        with col8:
            st.image("primary-emotions.jpg")

        st.subheader("Important Note: ")
        st.warning("🚨Currently SVM model is on maintenence!🚨")
        st.subheader("How to Predict a text? ")
        st.write("The following tutorial would help you through how to predict a text using a model called MUltinomialNB()")
        video_file = open("comercial_video.webm", "rb").read()
        st.video(video_file)
# --------------------------------------------------------------------------------FUCTIONS FOR PREDICTION-------------------------------------------------------------#

    elif choices == "Predict Emotions":
        st.subheader("Emotion Predictor-Emotions Text Classification")
        choices_1 = st.selectbox("Model Selector",choice)
        st.warning("🚨Currently SVM model is on maintenence!🚨",)
        def predict_emotions(text):
            if choices_1 == "SVM":
                results = clf_svm.predict(text)
                return results
            
            elif choices_1 == "GradientBoostingClassifier":
                results = clf_gradient.predict(text)
                return results
            
            elif choices_1 == "MultinomialNB":
                results = clf_multinomial_nb.predict(text)
                return results

            elif choices_1 == "KNearestNeighborsClassifier":
                results = clf_knn.predict(text)
                return results

        #------------------------------------------------------------------------FUNCTIONS FOR PROBABLITY---------------------------------------------------------------#
        def get_pred_score(text):
            if choices_1 == "SVM":
                results = clf_svm.predict_proba(text)
                return results
            
            elif choices_1 == "GradientBoostingClassifier":
                results = clf_gradient.predict_proba(text)
                return results
            
            elif choices_1 == "MultinomialNB":
                results = clf_multinomial_nb.predict_proba(text)
                return results

            elif choices_1 == "KNearestNeighborsClassifier":
                results = clf_knn.predict_proba(text)
                return results
        
        #--------------------------------------------------------------------------FORM-------------------------------------------------------------#
        with st.form(key="emotions_clf_form"):
            raw_text = st.text_area("Type Here: ")
            vec = CountVectorizer()
            submit_button = st.form_submit_button(label="Submit")
            X = vec.fit_transform(processed_text_df.processed_content)
            raw_text_ = vec.transform([raw_text])
        #---------------------------------------------------------------------SUBMIT FORM------------------------------------------------------------------#
        try:
            with open("data.pkl", "rb") as f:
                data_list = pickle.load(f)
        except EOFError:
            data_list = []
        except FileNotFoundError:
            data_list = []
        
        if submit_button:
            if raw_text == "":
                st.error("Bruh 😐 you gotta text something in the box before clicking submit!")
            
            else:
                st.subheader("Prediction Monitor: ")
                col1,col2 = st.columns(2)
            
            
                prediction = predict_emotions(raw_text_)
                probablity = get_pred_score(raw_text_)
                with col1:
                    st.success("Original Text")
                    st.write(raw_text)
                
                    if prediction == "joy":
                        st.success("Prediction")
                        st.write("Wow looks like your so happy 😀")

                    if prediction == "fear":
                        st.success("Prediction")
                        st.write("Chill! looks like your so frightened 😨")

                    if prediction == "anger":
                        st.success("Prediction")
                        st.write("Calm down looks like your so pissed off! 🤬")

                    if prediction == "sadness":
                        st.success("Prediction")
                        st.write("looks like your so sad, come on it will be fine 😭")

                    if prediction == "disgust":
                        st.success("Prediction")
                        st.write("ewww! looks like you are so disgusting 🤢")

                    if prediction == "neutral":
                        st.success("Prediction")
                        st.write("uhh,wait what am i supposed to say when you feel neutral 😐")

                    if prediction == "surprise":
                        st.success("Prediction")
                        st.write("Wow! looks like you are suprised 😲")
                with col2:
                    st.success("Prediction Probability")
                    proba_df = pd.DataFrame(probablity,columns=clf_svm.classes_)
                    st.write(proba_df.T)
                    proba_df_clear = proba_df.T.reset_index()
                    proba_df_clear.columns = ["emotions","probability"]
                
                st.subheader("Analysis Monitor: ")
                col3,col4 = st.columns(2)
                with col3:    
                    st.success("Chart")
                    fig = alt.Chart(proba_df_clear).mark_bar().encode(x = "emotions",y = "probability")
                    st.altair_chart(fig,use_container_width=True)

                with col4:
                    st.info("Saving the Data!")
                    data_list.append(({"Original Text":raw_text,"Model ":choices_1,"Result ":prediction}))
                    data_ = []
                    data_.append(({"Original Text":raw_text,"Model ":choices_1,"Result ":prediction}))
                
                
                    with open("data.pkl", "wb") as f:
                        pickle.dump(data_list, f,protocol=0)
                
                    st.write(data_)
# -------------------------------------------------------------------MONITOR--------------------------------------------------------------------------#
    elif choices == "Data Monitor":
        st.subheader("Data Monitor-Emotions Text Classification")
        st.subheader("Important Note: ")
        st.warning("🚨Currently SVM model is on maintenence!🚨")
        with open("data.pkl", "rb") as f:
            data_list = pickle.load(f)
            st.write(data_list)

#--------------------------------------------------------------------------__MAIN__-------------------------------------------------------------#
if __name__ == "__main__":
    main()