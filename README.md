# NLP-Learning Repository

Welcome to the **NLP-Learning** repository! This repository serves as a comprehensive learning resource for both basic and advanced **Natural Language Processing (NLP)** topics. It combines theoretical concepts with practical examples to give you hands-on experience.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Text Preprocessing](#text-preprocessing)
   - Data Cleaning
   - Tokenization
   - Stemming
   - Lemmatization
3. [Text Representation](#text-representation)
   - Bag of Words (BoW)
   - n-grams
   - TF-IDF
   - Word Embeddings
   - Transformer-Based Models
4. [Probabilistic Language Models](#probabilistic-language-models)
   - N-gram Models
   - Hidden Markov Models
   - Maximum Entropy Models
5. [Deep Learning Based Language Models](#deep-learning-based-language-models)
   - RNNs
   - LSTMs
   - Transformers
   - BERT
   - GPT
   - Llama
6. [Basic NLP](#basic-nlp)
   - Text Classification
   - Named Entity Recognition (NER)
   - Morphological Analysis
   - Part of Speech (POS)
   - Word Sense Disambiguation
   - Sentiment Analysis
7. [Advanced NLP](#advanced-nlp)
   - Question Answering
   - Information Retrieval
   - Recommendation Systems
   - Machine Translation
   - Text Summarization
8. [Models and Tools](#models-and-tools)
   - GPT & Gemini
   - Pipelines, Actions, Functions
   - RAG (Retrieval-Augmented Generation)
   - HuggingFace
   - OpenWebUI
   - LangChain
   - Ollama
9. [Usage](#usage)
10. [Contributing](#contributing)
11. [License](#license)

---

## [Introduction](./0-introduction/0.0-introduction.ipynb)

This section introduces the fundamental concepts of **Natural Language Processing (NLP)**. It provides an overview of the essential tasks, challenges, and techniques used in NLP. You'll learn about the applications of NLP in real-world scenarios, such as **machine translation**, **text generation**, and more.

---

## Text Preprocessing

**Text preprocessing** is a crucial first step in any NLP task. It involves preparing raw text data to make it suitable for machine learning models. Here are the key techniques:

- ### [Data Cleaning](1-text-preprocessing/1.0-text-preprocessing.ipynb)

  - This step involves removing unnecessary characters like special symbols, extra spaces, and formatting inconsistencies.

- ### [Tokenization](1-text-preprocessing/1.2-tokenization.ipynb)

  - Tokenization breaks the text into smaller units, such as words - or sentences, making it easier to process.

- ### [Stemming](1-text-preprocessing/1.3-stemming.ipynb)

  - Stemming reduces words to their base or root form (e.g., - "running" becomes "run").

- ### [Lemmatization](1-text-preprocessing/1.4-lemmatization.ipynb)

  - Lemmatization is similar to stemming, but it considers the context and converts words to their dictionary form (e.g., "better" becomes "good").

---

## Text Representation

Converting text into a format that machines can process is essential for any NLP task. Different techniques serve different purposes:

- ### [Bag of Words (BoW)](2-text-representation/2.1-bow.ipynb)

  - This method represents text as a set of word counts. While it's - simple and fast, it doesn't capture the context or semantic relationships between words.

- ### [n-grams](2-text-representation/2.3-n-gram.ipynb)

  - n-grams represent sequences of words. They can capture some - context and relationships between connected words, but they might require parameter tuning (e.g., n value).

- ### [TF-IDF](2-text-representation/2.2-tf-idf.ipynb)

  - TF-IDF emphasizes the importance of unique and relevant words by - considering both frequency and rarity in the document.

- ### [Word Embeddings](2-text-representation/2.4-word-embeddings.ipynb)

  - Word embeddings represent words as dense vectors in continuous space. They capture semantic similarities and relationships between words.

- ### [Transformer-Based Models](2-text-representation/2.5-transformer-based.ipynb)

  - Transformers (like BERT and GPT) use self-attention mechanisms to understand context and relationships in text, making them more powerful for complex NLP tasks.

---

## Probabilistic Language Models

These models calculate probabilities to predict the next word or sequence of words. Some popular models include:

- ### [N-gram Models](3-probabilistic-language-models/3.1-n-gram-models.ipynb)

  - N-grams are the simplest form of probabilistic models that predict the next word based on the previous words.

- ### [Hidden Markov Models](3-probabilistic-language-models/3.2-hidden-markov-models.ipynb)

  - HMMs use states and transitions between them to model sequential data, which can be useful for tasks like POS tagging or speech recognition.

- ### [Maximum Entropy Models](3-probabilistic-language-models/3.3-maximum-entropy-models.ipynb)

  - These models use probability distributions to estimate the likelihood of different outcomes in NLP tasks, ensuring no bias toward any specific outcome.

---

## Deep Learning Based Language Models

Deep learning models have revolutionized NLP, as they can learn complex patterns in large datasets. Some key models include:

- ### [RNNs (Recurrent Neural Networks)](4-deep-learning-based-language-models/4.2-rnn.ipynb)

  - RNNs are designed to handle sequential data by maintaining hidden states that capture information from previous time steps.

- ### [LSTMs (Long Short-Term Memory)](4-deep-learning-based-language-models/4.3-lstm.ipynb)

  - LSTMs are a type of RNN that mitigate the vanishing gradient problem, allowing them to capture longer-range dependencies in data.

- ### [Transformers](4-deep-learning-based-language-models/4.4-transformers.ipynb)

  - Transformers use self-attention to capture dependencies between all words in a sentence, making them more efficient for tasks like machine translation and text generation.

- ### [BERT](4-deep-learning-based-language-models/4.4-bert.ipynb)

  - BERT is a transformer-based model that focuses on understanding the context of words in a sentence. It's used for tasks like question answering and sentence classification.

- ### [GPT](4-deep-learning-based-language-models/4.5-gpt.ipynb)

  - GPT is a generative transformer model that excels at text - generation and can be fine-tuned for various NLP tasks.

- ### [Llama](7-models-and-tools/7.5-llama.ipynb)

  - Llama (Large Language Model Meta AI) is a family of transformer-based language models developed by Meta. It is widely used in NLP tasks such as text generation, classification, and more.

---

## Basic NLP

- ### [Text Classification](5-basic-nlp/5.0-text-classification.ipynb)

  - Text classification assigns labels to text based on its content. Popular techniques include Naive Bayes, SVMs, and deep learning models like LSTMs.

- ### [Named Entity Recognition (NER)](5-basic-nlp/5.1-named-entity-recognition.ipynb)

  - NER identifies entities like people, places, and organizations in text. It's used in information extraction tasks.

- ### [Morphological Analysis](5-basic-nlp/5.2-morphological-analysis.ipynb)

  - This involves analyzing words based on their structure, including processes like stemming and lemmatization.

- ### [Part of Speech (POS)](5-basic-nlp/5.3-part-of-speech.ipynb)

  - POS tagging assigns grammatical labels (e.g., noun, verb) to words in a sentence.

- ### [Word Sense Disambiguation (WSD)](5-basic-nlp/5.4-word-sense-disambiguation.ipynb)

  - WSD aims to determine which meaning of a word is activated by its context, using techniques like Lesk algorithm or cosine similarity.

- ### [Sentiment Analysis](5-basic-nlp/5.5-sentiment-analysis.ipynb)

  - Sentiment analysis determines the sentiment (positive, negative, or neutral) behind a piece of text, often used in product reviews and social media analysis.

---

## Advanced NLP

- ### [Question Answering](6-advanced-nlp/6.0-question-answering.ipynb)

  - Learn about extracting answers from a set of documents or a knowledge base, making it one of the most advanced NLP tasks.

- ### [Information Retrieval](6-advanced-nlp/6.1-information-retrieval.ipynb)

  - IR systems find relevant information from large datasets or databases based on a user's query.

- ### [Recommendation Systems](6-advanced-nlp/6.2-recommendation-systems.ipynb)

  - Recommendation systems suggest products, movies, or services based on user preferences and historical data.

- ### [Machine Translation](6-advanced-nlp/6.3-machine-translation.ipynb)

  - Machine translation automatically translates text from one language to another using NLP techniques.

- ### [Text Summarization](6-advanced-nlp/6.4-text-summarization.ipynb)

  - Text summarization creates a short summary of a long text while preserving its key information.

---

## Models and Tools

- ### [GPT & Gemini](7-models-and-tools/7.0-gpt-and-gemini/7.0.0-gpt-and-gemini.ipynb)

  - Learn about advanced language models like GPT and Gemini, which are used for various NLP tasks such as text generation, conversation, and summarization.

- ### [Pipelines, Actions, Functions](7-models-and-tools/7.1-pipelines-actions-functions/7.1.0-pipelines-actions-functions.ipynb)

  - This section teaches you how to set up an NLP pipeline and create reusable actions and functions for different tasks.

- ### [RAG (Retrieval-Augmented Generation)](7-models-and-tools/7.2-rag/7.2.0-rag.ipynb)

  - RAG combines information retrieval with text generation, allowing models to use external knowledge for tasks like question answering.

- ### [HuggingFace](7-models-and-tools/7.3-huggingface/7.3.0-huggingface.ipynb)

  - HuggingFace provides state-of-the-art models and tools for NLP - tasks. Learn how to use their pre-trained models for various NLP applications.

- ### [OpenWebUI](7-models-and-tools/7.4-openwebui/7.4.0-openwebui.ipynb)

  - OpenWebUI provides a web-based user interface for interacting with language models, enabling users to easily work with them through a browser interface.

- ### [LangChain](7-models-and-tools/7.5-langchain/7.5.0-langchain.ipynb)

  - LangChain connects language models with external data sources and helps build complex applications, such as question-answering systems, recommendation engines, or other data-driven NLP tasks.

- ### [Ollama](7-models-and-tools/7.6-ollama/7.6.0-ollama.ipynb)

  - OLLAMA allows easy integration of language models into applications, making it simple to deploy models in various production environments.

---

## Usage

Follow these steps to get started with this repository:

1. Clone the repository:
   `git clone https://github.com/erenisci/nlp-learning.git`

2. Install the necessary dependencies:
   `pip install -r requirements.txt`

3. Open the desired notebook and run the code.

---

## Contributing

We welcome contributions to this project! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to your branch (`git push origin feature-name`).
5. Open a pull request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
