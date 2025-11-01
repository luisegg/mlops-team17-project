# MLOPS Equipo 17 documentation!

## Description

This is the repository of the Team 17 for the MLOPS Project including the 3 phases, We try to analyze the energy comsupntion of the steel industry, trying to identify the factors and create a model to predict the compsuntion in order to improve the efficiency

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://mlops-team17-project/data/`.
* `make sync_data_down` will use `aws s3 sync` to recursively sync files from `s3://mlops-team17-project/data/` to `data/`.


