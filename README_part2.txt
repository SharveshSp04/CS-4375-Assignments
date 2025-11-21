# CS 4375 - Assignment 3 - Part II

## Tweets Clustering using K-means with Jaccard Distance

### Requirements
- Python 3.6+
- Required libraries: numpy, matplotlib, requests

### How to Run in Google Colab

1. **Upload the code to Google Colab:**
   - Copy the entire code block into a new Colab notebook
   - Or upload the .py file to Colab

2. **Run the code:**
   - The code will automatically:
     - Download the Health News Tweets dataset
     - Preprocess the tweets
     - Perform K-means clustering for K=[5, 8, 10, 12, 15]
     - Display results in table format
     - Plot SSE vs K graph
     - Show sample clusters

3. **Expected Output:**
   - Table showing K values, SSE, and cluster sizes
   - SSE vs K plot
   - Sample cluster information


### Libraries Used
- `numpy`: Numerical computations
- `matplotlib`: Plotting graphs
- `requests`: Downloading dataset
- `re`: Regular expressions for text processing
- `collections`: Data structures
- `random`: Random initialization
- `os`: File operations

### Algorithm Details

1. **Preprocessing:**
   - Remove tweet IDs and timestamps
   - Remove @mentions
   - Remove URLs
   - Convert hashtags to regular words
   - Convert to lowercase

2. **Jaccard Distance:**
   - Measures dissimilarity between tweet word sets
   - dist = 1 - |A ∩ B| / |A ∪ B|

3. **K-means Clustering:**
   - K-means++ initialization for better centroids
   - Custom centroid computation (tweet with minimum total distance)
   - Maximum 100 iterations with early convergence

### Note
The code automatically limits to 500 tweets for faster execution. You can modify the `max_tweets` parameter in the `load_tweets_from_file` call for larger datasets.
