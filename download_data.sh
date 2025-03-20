#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Download the dataset
echo "Downloading NYC Property Sales dataset..."
curl -L -o data/nyc-property-sales.zip \
  https://www.kaggle.com/api/v1/datasets/download/new-york-city/nyc-property-sales

# Unzip the dataset
echo "Extracting dataset..."
unzip -o data/nyc-property-sales.zip -d data/

# Clean up zip file
rm data/nyc-property-sales.zip

echo "Download complete. Dataset is in the data directory." 