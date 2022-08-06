import streamlit as st
from sklearn.preprocessing import LabelEncoder
from ctgan import CTGANSynthesizer
from sdv.metrics.tabular import KSTest
from sdv.metrics.tabular import LogisticDetection
import pandas as pd
import base64


def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()
    
    return f'<a href="data:file/txt;base64, {b64}" download="{download_filename}">{download_link_text}</a>'


le = LabelEncoder()
choices = ["Home", "Conditional GANs"]


def df_cat(arg):
    df = arg
    cols = df.columns
    cat_columns = df._get_numeric_data().columns
    to_delete_columns = list(set(cols) - set(cat_columns))
    
    st.warning("Text columns detected: " + str(to_delete_columns))
    st.warning("Converting text columns to numerical format and removing any NaN rows: ")
    
    for i in to_delete_columns:
        df[i] = le.fit_transform(df[i].astype('str'))
        
    mydf = df
    mydf = mydf.dropna()
    return mydf


menu = st.sidebar.selectbox("Menu", choices)
st.set_option('deprecation.showfileUploaderEncoding', False)


if menu == "Conditional GANs":
    st.title("CGANs")
    st.write("Upload a CSV file (preferrably not too large) and use the command bar to start the magic. We suggest you drop the columns which you don\'t need before uploading.")
    st.write("--------------------------------------")
    
    file_upload = st.sidebar.file_uploader("Upload CSV")
    if file_upload:
        df = pd.read_csv(file_upload)
        st.write(df.head())
        
        newdf = df_cat(df)
        st.write(newdf.head())
        
        amount = st.sidebar.text_input("How many rows of data you need to generate?")
        #name = st.sidebar.text_input("Enter name of column with Name: ")
        columns = st.sidebar.text_input("Enter name of discrete column: ")
        discrete_columns = [columns]
        num_epochs = st.sidebar.slider("Number of epochs to train: ", 20, 100)
        download = st.sidebar.checkbox("Download generated data?")
        generate = st.sidebar.button("GENERATE")
        
        if generate and download:
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
                
            st.write("This is taking some time...")
            ctgan = CTGANSynthesizer(epochs=num_epochs)
            ctgan.fit(newdf, discrete_columns)
            samples = ctgan.sample(int(amount))
            
            st.success("Successfully generated {} rows of data".format(str(amount)))
            st.balloons()
            st.write(samples)
            
            ktestacc = KSTest.compute(newdf, samples)
            ldacc = LogisticDetection.compute(newdf, samples)
            
            st.write("KTest accuracy: " + str(ktestacc*100) + "%")
            st.write("**Logistic Detection** accuracy: " + str(ldacc*100) + "%")
            
            tmp_download_link = download_link(samples, 'generated.csv', 'Click here to download your data!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
            
        elif generate:
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
                
            st.write("This is taking some time...")
            ctgan = CTGANSynthesizer(epochs=num_epochs)
            ctgan.fit(newdf, discrete_columns)
            samples = ctgan.sample(int(amount))
            
            st.success("Successfully generated {} rows of data".format(str(amount)))
            st.balloons()
            st.write(samples)
            
            ktestacc = KSTest.compute(newdf, samples)
            ldacc = LogisticDetection.compute(newdf, samples)
            
            st.write("KTest accuracy: " + str(ktestacc*100) + "%")
            st.write("**Logistic Detection** accuracy: " + str(ldacc*100) + "%")
            
elif menu == "Home":
    st.title("GAN based data generation")
    st.write("----------------------------------")
    st.subheader("You can upload any csv file on our app and generate upto 200 rows of synthetic data. Use our app to feed more data to your models and make them more accurate.")
    st.write("----------------------------------")
    st.header("Updates")
    st.write("**v1** of CGANS launched (3 days ago)")
    st.write("Visit the CGANS section from the menu and explore.")
    st.write('--------------------------------------')
    st.write("**v2** of CGANS launched (1 day ago)")
    st.write("Added conversion of text data to numerical data")
    st.write('--------------------------------------')
    st.write("**v3** of CGANS launched (today)")
    st.write("Added download functionality.")  