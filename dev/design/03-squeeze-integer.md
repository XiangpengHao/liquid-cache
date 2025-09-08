# Squeeze Integer


This doc describes how to implement `squeeze` for integer arrays.

Currently, we have LiquidPrimitiveArray, which is a bit-packed array.

It does not support squeeze, so on an eviction, will be entirely written to disk.

Instead, we'd like to support squeeze for integer arrays.

#### Motivating example

Let's say we have the following integer array:

```
[0, 32, 12, 90, 48, 368, 85, 13, 183]
```

The bit-width is determined by the max value, which is 368, which requires 9 bits.

To squeeze it, we squeeze the bit-width to 5 bits, which can now support up to 32. 

We reserve value=32 as the sentinel value, and the array becomes:

```
[0, 32, 12, 32, 48, 32, 32, 13, 32]
```

Now let's say we want to evaluate bit mask for all values i < 15, consider two cases:
1. if the value is smaller than 32, we know its actual value, which we can evaluate the predicate.
2. if the value is 32, we know its greater or equal to 32, which we know it's false.

What if we want to evaluate bit mask for all values i > 35?
1. for x < 32, we know the exact value. 
2. for x>= 32, we don't know the exact value, so we need to read from disk.

### Real design

To squeeze a LiquidPrimitiveArray, it is squeezable if the bit-width is great or equal than 10 (a intuition).
And we always reduce the bit-width by half.

To properly distinguish the fully in-memory and hybrid squeezed array, we need a LiquidPrimitive<Memory> and LiquidPrimitive<Disk>.

This part of the code is very similar to the LiquidByteViewArray<Memory> and LiquidByteViewArray<Disk> implementation.


#### Try evaluate predicate
By default, LiquidPrimitive<Memory> doesn't evaluate predicate, so will return `None`.

But for LiquidPrimitive<Disk>, we need to evaluate the predicate.
Using the example above, try to reduce the trip to disk as much as possible.

The supported predicate is: eq, not eq, lt, lt eq, gt, gt eq.

#### Get with selection
Get with selection will first check if the selected values are all full value, if so, return the data.
If any of the value is clamped, we need to get data from disk. 



## Quantize integer

A different way to squeeze is to quantize the integer array.

Let's say we have the following integer array:

```
[0, 32, 12, 90, 48, 368, 85, 13, 183]
```

The original array requires 9 bits.

If we want to quantize it to 5 bits, then we'll have 32 buckets, each bucket has 512/32 = 16 values, the bucket range is like:

```
[0, 32)
[32, 64)
[64, 96)
...
[336, 352)
[352, 368)
[368, 384)
...
[496, 512)
```

The original array becomes:

```
[0, 1, 0, 5, 3, 23, 5, 0, 11]
```

Let's say we want to find `66 < i < 72`, which belongs to the 3rd bucket.
Then we check if the quantized array contains 2, if not, we can evaluate the predicate, if yes, we need to read from disk. 

### Pros and cons

Squeeze:
- Pros: can recover the original value without reading from disk.
- Cons: most of the values are clamped, i.e., cutting the range by half will only save 1 bit. If we want to cut 10 bits, we shrink the range by 1024 times -- virtually every value is clamped.

Quantize:
- Pros: more tight approximation of the original value, good for predicate evaluation.
- Cons: none of the values can be recovered without reading from disk.

