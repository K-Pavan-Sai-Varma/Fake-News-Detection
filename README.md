Fake News Detection using NLP and Machine Learning

📌 Project Overview
This project implements a Fake News Detection system using Natural Language Processing (NLP) and a Machine Learning algorithm called Passive Aggressive Classifier. It classifies news articles as either FAKE or REAL based on the text content.


📁 Dataset
- I have used a custom dataset created for this project.
- The dataset includes labeled news articles (REAL and FAKE) in CSV format.
- It is not taken from any public GitHub repository.

📌 Dataset Fields:
- title: Name of the news article
- text: The news article content
- label: Classification (REAL or FAKE)

⚙️ Technologies & Libraries Used
- Python 3.8+
- pandas
- numpy
- scikit-learn (sklearn)
- matplotlib (for visualization)
- seaborn (optional, for enhanced graph visuals)


🚀 How to Run
1. Install the required libraries using pip:
   pip install pandas numpy scikit-learn matplotlib seaborn
2. Download or clone this repository.
3. Run the Fake_News_Detection.py script.
4. The program will output accuracy and a confusion matrix.
5. Graphs will be displayed showing fake and real article counts.


📊 Output
The system outputs:
- Accuracy of the model
- Confusion matrix showing prediction results
- Bar chart comparing real vs fake article counts
- Sample fake and real news articles

📃 License
This project is licensed under the MIT License
