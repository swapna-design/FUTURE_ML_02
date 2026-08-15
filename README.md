# FUTURE_ML_02
Automated support ticket classification system using NLP (NLTK) and machine learning — classifies tickets by category and predicts priority (High/Medium/Low) with TF-IDF feature extraction and Logistic Regression, including full text preprocessing and model evaluation.
# Support Ticket Classification System
**Future Interns — Machine Learning Internship | Task 2**

## 📌 Objective
Build a system that automatically classifies customer support tickets by category and assigns a priority level (High / Medium / Low), helping support teams respond faster and smarter.

## 🧰 Tools & Skills Used
- **Language:** Python (Google Colab / Jupyter Notebook)
- **Libraries:** NLTK, scikit-learn, pandas, numpy, matplotlib
- **Skills:** Text preprocessing & tokenization, NLP classification, TF-IDF feature extraction, priority logic, support analytics

## 📂 Project Structure
| File | Description |
|---|---|
| `support_tickets.csv` | Dataset of 1,200 support tickets with `ticket_text`, `category`, and `priority` columns |
| `Ticket_Classification.ipynb` / Colab cells | Full pipeline: text cleaning → tokenization → category classification → priority classification → evaluation → charts |
| `chart1_category_confusion.png` | Confusion matrix for category classification |
| `chart2_priority_confusion.png` | Confusion matrix for priority classification |

## 🔄 Workflow
1. **Text Preprocessing** — lowercased text, removed punctuation/numbers, tokenized with NLTK's `word_tokenize`, removed stopwords, and lemmatized each token to its root form.
2. **Feature Extraction** — converted cleaned ticket text into numerical features using **TF-IDF** (unigrams + bigrams, top 3,000 features).
3. **Category Classification** — trained a Logistic Regression model to classify each ticket into one of 5 categories: Billing, Technical Issue, Account Access, Product Inquiry, Feature Request.
4. **Priority Classification** — trained a second Logistic Regression model on the same TF-IDF features to predict priority (High / Medium / Low), based on urgency language present in the ticket text.
5. **Evaluation** — measured both models using accuracy, precision, recall, F1-score, and confusion matrices.
6. **Combined System** — wrapped both models into a single `classify_ticket(text)` function that takes a raw ticket and returns its predicted category **and** priority in one call — this is the actual deliverable a support team would plug into their workflow.

## 📊 Results
| Task | Accuracy |
|---|---|
| Category Classification | ~99–100% |
| Priority Classification | ~99–100% |

**Note:** Accuracy is very high because the training data is synthetically generated from clean templates for demonstration purposes. Real-world tickets (typos, mixed topics, ambiguous urgency, sarcasm) would bring this down — a realistic production target is typically 80–90% for category and 70–85% for priority. This is called out here for transparency.

## 💡 Business Insights
- **Technical Issue** and **Account Access** tickets skew toward High priority — these block users from working and should be routed to support agents first.
- **Product Inquiry** and **Feature Request** tickets skew toward Low priority — informational, not blocking.
- Keyword/phrase patterns like "urgent," "ASAP," "blocking my team," and "production system" are strong priority signals the model picks up automatically.
- Automating this triage step means support teams spend less time manually reading and sorting tickets, and high-priority issues surface immediately instead of sitting in a queue.

## 🚀 How to Run
1. Open [Google Colab](https://colab.research.google.com/drive/1-UnUezpjBNG-S8b5TvoYViix8TSc3B3U#scrollTo=3sACGgJXj7NN).
2. Upload `support_tickets.csv` (or your own dataset with `ticket_text`, `category`, `priority` columns).
3. Run the notebook cells in order:
   - Install & import NLTK
   - Load dataset
   - Clean & tokenize text
   - Train category classifier
   - Train priority classifier
   - Test the combined `classify_ticket()` function on new tickets
   - View confusion matrices and charts

## ✅ Deliverable
A working ticket classification system that takes raw customer support text and automatically returns both a **category** and a **priority level**, submitted as part of the Future Interns Machine Learning Internship (Task 2).

