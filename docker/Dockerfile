# Use the latest Jupyter Notebook image
FROM jupyter/datascience-notebook:latest

# Set the working directory in the container
WORKDIR /home/jovyan/work

# Install jupytext here
RUN pip install jupytext

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run jupyter notebook when the container launches
CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]

