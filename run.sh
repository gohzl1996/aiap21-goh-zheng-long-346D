#!/usr/bin/env bash
# ElderGuard Analytics pipeline runner
# Navigation: cd /mnt/c/Users/longs/OneDrive/Desktop/AIAP/aiap21-goh-zheng-long-346D

# WSL commands to use if need to set up python environment to run this file:

# sudo apt update
# sudo apt install python3.12-venv -> If you’re using a different Python version, adjust accordingly (e.g., python3.10-venv).
# rm -rf venv (Removes python environment)
# python3 -m venv venv (Recreates python environment)
# source venv/bin/activate
# ./run.sh

set -e  # exit immediately if a command fails

#echo "Setting up Python virtual environment..."
#python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

#echo "Upgrading pip..."
#pip install --upgrade pip

#echo "Installing dependencies from requirements.txt..."
#pip install -r requirements.txt

echo "Running ElderGuard pipeline..."
python -m src.main

echo "Done."
