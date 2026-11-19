# Missing People Face Detection
This project explores how facial recognition can support missing-person searches by comparing uploaded photos with reference images in a sample dataset. It uses InsightFace and CNN-based deep-learning models to detect and compare faces, surfacing potential matches for human review rather than confirming identity.

## How the Service Could Work

People who may be at risk of becoming lost, such as people living with dementia, could register with the organization in advance. With appropriate consent, the organization would keep recent photos and contact details for a trusted family member or caregiver.

If someone is reported missing, or a found person cannot be identified, authorized responders could compare an available photo with enrolled profiles. Potential matches could guide search and reconnection efforts and should be confirmed before details are shared.

This demo illustrates photo comparison using a local sample dataset; enrollment and secure profile management are part of the broader concept.

## Running the Demo

### Step 1: Install the Model
- Download the model and add it to the `demo` folder. You can find the model <a href="https://drive.google.com/file/d/1FPldzmZ6jHfaC-R-jLkxvQRP-cLgxjCT/view" target="_blank">here</a>
.

### Step 2: Create virtual environment 

1. Create the python virtual environment using:
   ```bash
   python3 -m venv venv-name

2. Activate it using:
   ```bash
   source venv-name/bin/activate


### Step 3: Set Up the Demo

1. Install the dependencies using `pip` from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt

2. Navigate to the `demo` folder:
   ```bash
   cd demo

3. Run Streamlit on the main script:
   ```bash
   streamlit run main.py
   ```

4. In the Streamlit page, click **Upload an image** and choose a sample photo from the project's `random-people/` folder to try the demo.