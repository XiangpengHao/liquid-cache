-- Q4 - top 3 post veterans
SELECT
    variant_get(data, 'did', 'string') AS user_id,
    MIN(TO_TIMESTAMP_MICROS(variant_get(data, 'time_us', 'bigint'))) AS first_post_ts
FROM bluesky
WHERE variant_get(data, 'kind', 'string') = 'commit'
  AND variant_get(data, 'commit.operation', 'string') = 'create'
  AND variant_get(data, 'commit.collection', 'string') = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY first_post_ts ASC
LIMIT 3;

