# StockAnalyzer

Analyzes NYTimes headlines using LSTM and other language processing techniques, and predicts gain in a certain stock on the NYSE

To use, must take three steps:
1. Obtain Quandl and NYTimes API keys and put them in a file called keys.json of the form:
  {
      "quandlkey":"<actualquandlkey>",
      "nytimeskey":"<actualnytimeskey>"
  }
2. Make directory 'Data'
3. Download the Google word2vec pretrained model and put it in the directory. Can be found [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
