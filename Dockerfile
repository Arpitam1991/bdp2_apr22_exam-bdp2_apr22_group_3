# Use Python 3.10 as the base image
FROM python:3.10

# Make a directory inside container
RUN mkdir app

# Setting working directory inside the container
WORKDIR /app

# Setting the global index URL for pip
RUN pip config set global.index-url https://pypi.org/simple/

# Copying the saved model, code for the model and the web application files
COPY sentvenv/src/main.py .
COPY sentvenv/src/app.py .
COPY sentvenv/src/model.h5 sentvenv/src/.

#Requirements
COPY requirements.txt .

# Activating the virtual environment and installing dependencies
RUN python3 -m venv sentvenv && \
    /bin/bash -c "source sentvenv/bin/activate && pip install --timeout=100 --no-cache-dir -r requirements.txt"

# Expose port 8050 for Dash app
EXPOSE 8050

# Defining environment variable
ENV NAME World

# Setting the command to run the model
CMD ["/bin/bash", "-c", "source sentvenv/bin/activate && python app.py"]
