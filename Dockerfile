# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY ./src ./src
COPY ./data ./data

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Command to run the application
# Note: This is a simple way to run both. For production, you might use a process manager.
# First, run the data processing script.
# Then, start the API server in the background and the Streamlit app in the foreground.
CMD ["sh", "-c", "python src/data_processing.py --input_filepath data/customer_shopping_data.csv --output_filepath data/processed_customer_data.parquet && uvicorn src.api:app --host 0.0.0.0 --port 8000 & streamlit run src/dashboard.py --server.port 8501 --server.address 0.0.0.0"]