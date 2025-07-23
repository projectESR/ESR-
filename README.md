Automated Blood Grouping System
This project is a web-based application that uses machine learning and computer vision to automatically determine blood types from an image of a test card.

Features
Web Interface: Modern, user-friendly interface for uploading images.

Automated Analysis: The backend, powered by Python and Flask, automatically splits the test card image into sections.

ML-Powered: Uses a trained Scikit-learn model to detect agglutination based on texture features (GLCM).

Fallback System: Includes a rule-based system for analysis if the ML model is not available.

Detailed Results: Displays the final blood type and shows the original and segmented images for each test section.

How to Run
This application can be run locally, in GitHub Codespaces, or using Docker. Please refer to the full guide for detailed instructions.
