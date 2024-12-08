# Missing People Face Detection
This is a project using InsightFace model to detect missing people which is based on the CNN deep learning algorithm.

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