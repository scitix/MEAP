#!/bin/bash

# External parameter for the Python script
PYTHON_SCRIPT=$1

# Run the model with the provided Python script
lightning run model --devices=1 --accelerator=cuda $PYTHON_SCRIPT
