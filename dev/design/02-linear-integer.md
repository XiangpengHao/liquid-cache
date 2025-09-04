# Integer array through linear regression


### Motivating example

Let's say we have the following integer array:

```
[0, 12, 32, 48, 64, 85, 90]
```

There're two ways to compress this array:
1. Direct bit-packing.
2. Delta encoding, then bit-packing.

Approach 1 is not great because the bit-width depends on the max value of the array.
Approach 2 is not great because it disallow random access.

Instead, we can use a linear model to compress the array.

Let's say we have the following linear model: y = mx + b.

Where `m` is the slope and `b` is the intercept, we assign `m = 15` and `b = 0`.

Then if using this model, we get the array:
[0, 15, 30, 45, 60, 75, 90]

To recover the original array, we need to store an error term:
[0, -3, 2, 3, 4, 5, 0]

Now we can use bit-packing to store the error term.

The idea is that error term is much smaller than the original values, so we will likely use much narrower bit-width. 

### Real design

Given a integer array from Arrow, we want to compress it using a linear model.

Option 1:
We simply calculate the slope and intercept using simple approach:
1. find the min and max value of the array.
2. assuming min is at location 0, and max is at location `n`, then the slope is `(max - min) / n`, and the intercept is `min`.

Option 2:
We use a linear regression model to calculate the slope and intercept.
Later note: using linear regression is wrong, because we care about the largest error, not the sum of l2 distance of errors. Also, using linear regression does not guarantee the error term is smaller than the max value of the original array. 


Once we have the linear model, we need to compute the error term.

The error term is a signed integer array, initially using i32. We evaluate every element in the array and store the error term.

Once we have the error term, we can use the existing bit-packing, i.e., `LiquidI32Array` to compress it, which will automatically use the narrowest bit-width and handle negative values.


### Study

We mainly concerns about the compression ratio.

To do this, we'll create a study just like `study/cache_storage.rs`, which will:
1. read integer columns from hits.parquet dataset.
2. compress the integer columns using the linear model, and the existing bit-packing approach.
3. Report: 1. arrow data size, 2. existing bit-packing size, 3. linear model size.
4. Report: 1. existing bit-packing compress/decompress speed, 2. linear model compress/decompress speed.

