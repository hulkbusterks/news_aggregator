# Use an official Python runtime as a parent image
FROM python:3.11

# WORKDIR /app


# Install required C++11 libraries and ca-certificates
# RUN apt-get update -qq \
#       && apt-get install -y \
#       build-essential \
#       python3-dev \
#       ca-certificates \
#       && apt-get clean \
#       && rm -rf /var/lib/apt/lists/*
# Install dependencies for Selenium and Chrome
# RUN apt-get update && apt-get install -y \
#     wget \
#     unzip \
#     curl \
#     gnupg \
#     libnss3 \
#     libgconf-2-4 \
#     libxi6 \
#     libxcursor1 \
#     libxrandr2 \
#     libxss1 \
#     libxtst6 \
#     fonts-liberation \
#     xdg-utils \
#     libatk-bridge2.0-0 \
#     libgtk-3-0 \
#     --no-install-recommends && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y wget unzip && \
#     wget https://dl.google.com/Linux/direct/google-chrome-stable_current_amd64.deb && \
#     apt install -y ./google-chrome-stable_current_amd64.deb && \
#     rm google-chrome-stable_current_amd64.deb && \
#     apt-get clean
# # Get secret HF_HOME and output it to /test at buildtime
# RUN --mount=type=secret,id=HF_HOME,mode=0444,required=true \
#    cat /run/secrets/HF_HOME > /test
# Get secret GROQ_API_KEY and output it to /test at buildtime
RUN --mount=type=secret,id=GROQ_API_KEY,mode=0444,required=true \
   cat /run/secrets/GROQ_API_KEY > /test

RUN useradd -m -u 1000 user
USER user
# ENV PATH="/home/user/.local/bin:$PATH"
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# WORKDIR /app
WORKDIR $HOME/app
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# COPY --chown=user . /app
COPY --chown=user . $HOME/app


# # Make port 5000 available to the world outside this container
# EXPOSE 8000



# Run main when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]