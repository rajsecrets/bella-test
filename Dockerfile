# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Set environment variables
ENV GEMINI_API_KEY=your_gemini_api_key
ENV SERP_API_KEY=your_serp_api_key

# Expose the port the app runs on
EXPOSE 8000

# Run the app
CMD ["chainlit", "run", "app.py", "--port", "8000"]