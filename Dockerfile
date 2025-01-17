Set-Content -Path Dockerfile -Value @"
# Step 1: Use an official Python image
FROM python:3.8-slim

# Step 2: Set a working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements.txt into the container
COPY requirements.txt /app/

# Step 4: Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of the app code into the container
COPY . /app/

# Step 6: Expose the port that Streamlit will use
EXPOSE 8501

# Step 7: Set the command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
"@

