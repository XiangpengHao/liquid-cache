String fingerprint helps to determine if a pattern is present in a string.

Reference: https://arxiv.org/pdf/2507.10391

It divides the letter space into n buckets, each bucket contains a few letters.
Then for the original string, we check each bucket, and if the bucket's letter is present in the string, we mark it as 1, otherwise 0. 
So we get a bit vector of length n, A.

To know if a pattern is present in the string, we do the same for the pattern, and get another bit vector of length n, B.

Then we check if 1s in B are all present in A. If not, the pattern is not present in the string. If yes, the pattern is probably present in the string.

## Example

bucket 1: a, l, u
bucket 2: v, ., o, w
bucket 3: e, n, t
bucket 4: d, -, b

String: nutella -> 1 0 1 0


Pattern 1: google -> 0 1 0 0 -> not in the string
Pattern 2: ntu -> 1 0 1 0 -> probably in the string

## Implementation

To make it simple, we only consider bytes, not letters.
Each byte has 256 values, and we distribute them into n buckets.
We distribute them through round-robin, i.e., value 0 goes to bucket 0, value 1 goes to bucket 1, value 2 goes to bucket 2, and so on.

Then for a StringArray (or BinaryArray), we compute a fingerprint for each value.

To test whether a pattern is present in the string, we compute a fingerprint, and check if the 1s in the fingerprint are all present in the string's fingerprint.

For each parameter n, we measure:
1. For a given pattern, how many strings can be determined to not present in the string array.
2. For ones that are determined to be present, how many of them are false positives.

## Study

Bucket sizes: 4, 8, 12, 16, 20, 24, 28, 32.
String data: the URL, Title, Referer columns from clickbench (benchmark/clickbench/data/hits.parquet).
Patterns: google

## Optimizations

The bucket assignment matters.
Instead of using a predefined round-robin (or contiguous range), we can build a custom mapping table. 

This is done by:
1. Sampling the fist few strings (e.g., 100) in the column, and build a histogram of the bytes.
2. Given a pattern, e.g., "google", we know there're four unique bytes {g, o, l, e}.
3. We assign the four bytes to four different buckets.
4. Sort remaining bytes by their frequency, and assign them to the current lowest frequency bucket.

## Results

| Column | Gram | Mapping | n | Rows | Nulls | Filtered Out | % | Candidates | % | False Pos | % | Actual Present | % |
|--------|------|---------|---|------|-------|--------------|---|------------|---|-----------|---|----------------|---|
| URL | One | RoundRobin | 32 | 1,000,000 | 0 | 605,825 | 60.58% | 394,175 | 39.42% | 394,133 | 99.99% | 42 | 0.00% |
| Title | One | RoundRobin | 32 | 1,000,000 | 0 | 877,755 | 87.78% | 122,245 | 12.22% | 122,242 | 100.00% | 3 | 0.00% |
| Referer | One | RoundRobin | 32 | 1,000,000 | 0 | 628,936 | 62.89% | 371,064 | 37.11% | 276,287 | 74.46% | 94,777 | 9.48% |


## How it helps LiquidCache
1. Accelerates substring search evaluation.
2. Reduces number of io, if the entire batch can be filtered out.
3. Reduces number of strings to be decompressed, if the batch of strings can't be filtered out.

