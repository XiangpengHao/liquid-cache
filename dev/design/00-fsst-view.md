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
                                                                  
──────────────────────────────────────────────────────────────────
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
