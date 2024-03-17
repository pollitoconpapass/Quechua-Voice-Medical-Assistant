# Cuzco Quechua Voice Medical Assistant
### Definition 
A Voice assistant where you can talk and dicuss about medical stuff in Cuzco Quechua

### Usage
1. Install all the dependencies

        pip install -r requirements.txt
2. Create a Service Credential in Cloud Translate API and append the JSON file into the model.py (ln.14)

    Get it here: https://cloud.google.com/translate/docs/authentication
    
3. Download a LLAMA-2 model and append it to the project: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML

4. Run the `ingest.py` file to create the FAISS db

        python ingest.py
5. Run the endpoints 

        python recognition.py
6. In another terminal run the app.py and enjoy!

        python app.py
