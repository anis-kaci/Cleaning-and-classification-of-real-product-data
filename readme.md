
# Real Security Products Catalog Cleaning and Classification

## Project Overview
This project aims to clean and classify a company extensive catalog of security products encompassing hardware (such as switches, firewalls), software (including antivirus, VPNs), and services (varied support durations). The catalog data originates from diverse sources within different business units and countries, resulting in inconsistencies in conventions and formats.

## Objectives
- **Data Cleaning:** Standardize the catalog by resolving inconsistencies, removing duplicates, handling missing values, and unifying the representation of product information.
- **Classification:** Utilize the Random Forest algorithm to classify and organize products into relevant categories based on their attributes and descriptions.

## Data Sources
The catalog data was obtained as a compilation from various sources of the company.

## Methodology
### Cleaning Process
1. **Data Gathering:** Downloaded the entire catalog encompassing diverse security products.
2. **Data Inspection:** Identified inconsistencies, missing values, and duplicates.
3. **Cleaning Operations:** Utilized Python and Pandas for data handling, employed various cleaning techniques to standardize and harmonize the information.

### Classification Process
1. **Feature Engineering:** Extracted relevant attributes from product descriptions and specifications.
2. **Model Selection:** Employed Random Forest, which is an ensemble learning method that combines multiple decision trees to make predictions.This algorithm is suitable for classification tasks.
3. **Model Training:** Utilized Python's scikit-learn library for training and optimizing the Random Forest classifier.
4. **Model Evaluation:** Assessed classifier performance and refined model parameters for improved accuracy.

## Results
- **Cleaned Catalog:** The catalog has undergone comprehensive cleaning, resulting in a standardized representation of RealSec LLC's security products.
- **Classification Outcome:** The Random Forest model effectively categorized products into hardware, software, and service segments, aiding in organizing the extensive catalog for improved usability and navigation.

## Next Steps
- **Continuous Improvement:** Ongoing efforts to refine the cleaning process for better standardization and explore advanced classification techniques.
- **Integration:** Integrate the cleaned and classified catalog into the company for improved accessibility and customer experience.
