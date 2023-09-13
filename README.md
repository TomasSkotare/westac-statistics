# Package for parsing the Swedish parlimentary proceedings
The proceedings can be found [here](https://github.com/welfare-state-analytics/riksdagen-corpus)

This can create a Pandas dataframe containing all speeches found, along with who spoke and when.

The usual way to use this method is in combination with Jupyter notebooks that are not included as part of this package.

# Docker-compose file

The github repository includes a docker-compose file along with necessary files for execution.

This includes:
- Notebooks in .py format (must be converted using jupytext)
- Stop words for the notebooks
- docker-compose files for execution
- Scripts for converting between notebooks and py files

## Docker execution
Note that the notebooks expects to have access to the riksdagen-corpus files, so they must also be added by the user in the directory `riksdagen-corpus` in the root of this repository.

To then start the file, simply go to the `docker` directory and execute the command `docker-compose up -d` to start the container. The default listening port is `9006` instead of the default 8888, so if the notebook is started on the local machine [this link](http://localhost:9006) should work

During startup, the notebooks in `.py` format will be automatically converted to jupyter notebook format, but no files will be overwritten. If the user wishes to redo the conversion, delete the appropriate notebook files and run the script again manually or restart the container.

Similarly, if changes should be converted back to `.py` format from notebook format, use the script `notebook_to_py.sh` inside the docker container.




