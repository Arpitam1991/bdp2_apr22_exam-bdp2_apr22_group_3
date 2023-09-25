FROM python:3.10

# Make a directory inside container
RUN mkdir app

# Setting working directory inside the container
WORKDIR /app

# Setting the global index URL for pip
RUN pip config set global.index-url https://pypi.org/simple/

# Copying the saved model, code for the model and the web application files
COPY main.py .
COPY app.py .
COPY model.h5 .
COPY requirements.txt .

# Activating the virtual environment and installing dependencies
RUN python3 -m venv myenv && \
    /bin/bash -c "source project_sentiment/bin/activate && pip install --timeout=100 --no-cache-dir -r requirements.txt"

# Expose port 8050 for Dash app (informative only, remember to map the port when running the container)
EXPOSE 8050

# Defining environment variable
ENV NAME World

# Setting the command to run the model
CMD ["python", "app.py"]
