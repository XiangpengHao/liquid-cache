# FSSTView in LiquidCache

## Current string representation

```
Dictionary(u16)    FSSTArray(BinaryArray)                   
   ┌────────┐     ┌─────────────────────────────────────┐   
   │ ┌────┐ │     │ Offset(i32)Nulls   FSST Buffer      │   
   │ │ 12 │ │     │   ┌─────┐ ┌─────┐  ┌──────────────┐ │   
   │ └────┘ │     │   │     │ │     │  │              │ │   
   │ ┌────┐ │     │   │     │ │     │  │              │ │   
   │ │ 37 │ │     │   │     │ │     │  │              │ │   
   │ └────┘ │     │   │     │ │     │  │              │ │   
   │ ┌────┐ │     │   │     │ │     │  │              │ │   
   │ │ 42 │ │     │   │     │ │     │  │              │ │   
   │ └────┘ │     │   │     │ │     │  │              │ │   
   │ ┌────┐ │     │   │     │ │     │  │              │ │   
   │ │ 17 │ │     │   │     │ │     │  │              │ │   
   │ └────┘ │     │   │     │ │     │  │              │ │   
   └────────┘     │   └─────┘ └─────┘  └──────────────┘ │   
                  └─────────────────────────────────────┘   
```
## New string representation

```
 Nulls  DictionaryView(u64)   Offset │   FSST Buffer           
 ┌───┐  ┌────────────────┐   ┌─────┐ │  ┌─────────────────────┐
 │   │  │┌──┐┌──────────┐│   │ i32 │ │  │                     │
 │   │  ││12││prefix(6b)││   │     │ │  │                     │
 │   │  │└──┘└──────────┘│   │     │ │  │                     │
 │   │  │┌──┐┌──────────┐│   │     │ │  │                     │
 │   │  ││37││prefix(6b)││   │     │ │  │                     │
 │   │  │└──┘└──────────┘│   │     │ │  │                     │
 │   │  │┌──┐┌──────────┐│   │     │ │  │                     │
 │   │  ││42││prefix(6b)││   └─────┘ │  │                     │
 │   │  │└──┘└──────────┘│           │  │                     │
 │   │  │┌──┐┌──────────┐│           │  │                     │
 │   │  ││17││prefix(6b)││           │  │                     │
 │   │  │└──┘└──────────┘│           │  │                     │
 └───┘  └────────────────┘           │  └─────────────────────┘
                                     │                         
                            In-memory│Disk                     
```

TLDR: 
1. dictionary keys are stored as a 2-byte key index and a 6-byte prefix.
2. Offset and nulls are stored in memory.
3. FSST buffer is stored on disk.

Design decisions:
1. The strings in the FSST buffer are unique, i.e., if two strings are different, their dictionary keys are different, and vice versa.
2. There's only one FSST buffer, this avoids the need to track buffer ids as in StringView representation in Arrow. 
3. Offset refers to the string offset in the FSST buffer, it use Arrow's offset buffer.
4. Nulls refers to the null bit of the DictionaryView.
5. DictionaryView consumes 8 bytes, with fixed 6-byte prefix. 
6. Everything but FSST buffer is stored in memory.

Questions:
1. should we bit-pack the offsets?
2. should we merge the null bits into DictionaryView? Maybe read this paper: https://dl.acm.org/doi/pdf/10.1145/3662010.3663452


## Design notes

### Prefix are uncompressed
The whole purpose of the prefix is to skip the symbol table thing.

### Prefix is 6 bytes
It can be 14 bytes, but that's probably too much.

### DictionaryView is different than StringView 
Each view in StringView has 16 bytes, this is too large.

FSSTView shrinks it to 8 bytes by:
1. remove buffer id
2. remove offset and len by storing dictionary offset in a separate field, and use dictionary index to get the offset and len.

### Use prefix to skip decompression

When comparing FSSTView with a string needle, we can skip the decompression by using the prefix: first check if the 6-byte prefix is enough to determine the result, if not, then decompress the string for comparison.

Design discussion:
- Sometimes it's faster to decompress the entire array and then do the comparison. But when?


## Engineering details

### Fuzz test

1. Use cargo-fuzz to test the FSSTView implementation.
2. We use [structured fuzzing](https://rust-fuzz.github.io/book/cargo-fuzz/structure-aware-fuzzing.html#structure-aware-fuzzing) to generate array of strings and 10 `compare_with` operations.
3. We test the following functions:
- Roundtrip from and to arrow StringArray.
- The `compare_with` function. We test that our `compare_with` function is equivalent to the Arrow's equivalent function.

### Evict to disk

FSSTView can be evicted to disk, but we only evict the FSST buffer and keep the DictionaryView in memory, this allows most of the time to avoid decompression and IO.

To do this, we need to change the `fsst_buffer` to be an enum, with two variants:
1. `InMemory(FsstArray)`
2. `OnDisk(PathBuf)`

We will need to add two functions:
1. `evict_to_disk`: evict the FSST buffer to disk, and keep the DictionaryView in memory. the enum will change from `InMemory` to `OnDisk`.
2. `load_from_disk`: load the FSST buffer from disk, and keep the DictionaryView in memory. the enum will change from `OnDisk` to `InMemory`.

The above two functions will need to be thread-safe, so a `std::sync::RwLock` is needed.

When we need to read from `fsst_buffer`:
1. If it's `InMemory`, we can read from it directly.
2. If it's `OnDisk`, we read it from disk, do the work, and drop the in-memory data, i.e., **no promotion policy**.
