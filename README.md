# RedditSentimentClassifier
Classify Crypto Comments from Reddit into POS/NEG sentiment

ML Model can be downloaded from this below link - 
 - https://drive.google.com/file/d/1YZ2HaH3_kzce-r0lRs-Tw9TUb5kUa9zc/view?usp=share_link
 - It's a zip file name checkpoint-2400.zip
 - Please unzip it and put it under the rootOfProject/model/checkpoint-2400

Analysis & Observations -

    - Although I've tried my best to discuss/explain my approaches and reasoning behind each one of them, I'm adding a brief high-level summary of all the notebooks below.

    1_Explo_&_PreProc-RedditCrypto.ipynb
        This notebook mainly contains my initial explorations as well as cleaning utilities. 
        Some of the highlights here are looking at Word Clouds and Top/Least Frequent words in Vocab.

    2_Basic_Sentiment_Analyzer.ipynb
        Just for sake of establishing the baseline I've shown in this one, results from VaderSentimentAnalyzer.
        Also, tells us why we need the ML model.

    3_1_ML_Classifier_UNI.ipynb
        Here I've tried training 5 basic ML models on a provided dataset of Reddit.
        Featurization of input text has been approached via 2 techniques, CountVectorizer, TFIDFVectorizer (Unigram Models).
        For more details, Please have a look at SECTIONS: Configuration, Observations from Results.

    3_2_ML_Classifier_UNI_BI_TRI.ipynb
        Here, I've tried the approach of using a model that uses UNIGRAMs/BIGRAMs/TRIGRAMs while building Feature Vectors.
        For more details, Please have a look at SECTIONS: Configuration, Observations from Result.

    ********* In notebooks, 4, 6, 7_2 ********* 
    - I'm trying to leverage the external dataset to see if I can get better results while still using the same featurizations techniques & classical ML models.
    
    4_Explo_&_PreProc-TWNS.ipynb
        These two notebooks mainly do the preprocessing for external Dataset (Twitter Financial News Sentiment Classification)
        In 4 specifically, I clean and transform the data into supportable format. Please have a look @ Data Description Section.

    6_Explo_&_PreProc-RedditCrypto_&_TWNS.ipynb
        In 6 specifically, I'm just combining this newly sourced data with the Reddit Dataset. Some stats are also mentioned in the notebook.

    7_2_ML_Classifier-RedditCrypto_&_TWNS.ipynb
        This notebook simply uses the combined dataset to train the model to check if results are improving.
        Have a look at SECTIONS: Idea, Configuration, Observation from Results

    - Unfortunately, none of the Classical ML models with TFIDF like featurization methods yields great results.
    - The Essential problem lies with the size of the Reddit Dataset.
    - Transfer learning with BAG/TFIDF schemes of vectorization didn't turn out to be helping.
    - Tried a couple of tuning techniques as well.
    - Best results I got with the ML Model on the REDDIT Dataset is around ~67.0% of accuracy.

    ********* Moving into DeepLearning Space *********

    8_0_PreProcessing-StockTwits-crypto.ipynb
        Onboarding another dataset called StockTwits-Crypto to be later used for fine tuning.
        Have a look at SECTIONS: Idea, Data Description.

    8_1_DL_Classifier-RedditCrypto-BaseLine.ipynb
        In this one, I'm using distilbert-base-uncased-finetuned-sst-2-english model to establish the new baseline for DL based approaches.
        Simple Baseline model yields ~72% Accuracy on Reddit Dataset.
        For more in-depth commentary, please have a look at SECTIONS: Configuration, Observation from Results.

    8_2_DL_Classifier-StockTwits-FineTuning_GoogleColab.ipynb
        Here, I finetune the baseline model on externally sourced StockTwits-Crypto Dataset (100k) samples via google colab.
        Please have a look at SECTIONS: Sections: Data Description, Points to be Noted, Configurations.
        In this whole finetuning on externally sourced dataset, I'm never showing the actual Reddit Crypto Comments Sentiment Dataset to model.
        My hunch behind not using Reddit Dataset for fine tuning at all is that it won't make any difference as size of Reddit Dataset is very small.

    8_3_DL_Classifier-RedditCrypto-FineTuned_GColab.ipynb
        Show Time, this is the most important notebook.
        Here, I simply test the Reddit Dataset on a fine tuned model from an earlier 8_2 notebook.
        It turned out that the Model now yields ~85% of accuracy.
            F1 Score for Negative Class: 0.85
            F1 Score for Positive Class: 0.86
        For more detailed analysis, please go through SECTIONS: Configurations, Observations from Results, Areas of Further Improvements

    - I've also written a simple inference server script, added under the project root.