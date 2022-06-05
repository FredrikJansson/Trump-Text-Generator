# Trump-Text-Generator
Uses RNN and LSTM for generating texts using data from his Twitter.

## Dataset
The dataset used is provided from The Trump Archives (https://www.thetrumparchive.com/), which in turn keeps track of all of Trumps tweets since 2016 (whether it has been deleted since).

The entire archive can be downloaded through a link in its FAQ page, and is later made into a proper dataset for the script through the use of `fix_dataset.py`. This script expects the JSON version.

You can manually choose how many tweets you want in the final dataset by providing a parameter; `python3 fix_dataset.py [NUMBER OF TWEETS]`. E.g., `python3 fix_dataset.py 1000` will fetch the first 1000 tweets. There is no randomizing so it will just take tweets in order.

A valid parameter for max number of tweets would be 0-positive infinity, but it would stop once it reaches the end. Any number zero or less is the same as reading all tweets.

## Running
### Create dataset.
1. Download the JSON archive file from the FAQ page of Trump Archives.
2. Place it in the same folder as `fix_dataset.py` and rename it to `tweets.json`.
3. Run `python3 fix_dataset.py` or alternatively provide a max number of tweets by running `python3 fix_dataset.py [number_of_tweets]`.

### Creating the model.
1. Installing requirements using `pip3 -r requirements.txt`.
2. Running the main script, `python3 main.py -t/--train`.

### Querying the model.
The model can be queried in two different ways. Either through a single query, or continuously queried til you send in an empty string.
#### Single Query
1. `python3 main.py [query]`

#### Continued Query
1. `python3 main.py -a`
2. When "- " is prompted, you can type in starting text.