-- Q4 - top 3 post veterans
SELECT
    variant_get(data, 'did', 'Utf8') AS user_id,
    MIN(TO_TIMESTAMP_MICROS(variant_get(data, 'time_us', 'Int64'))) AS first_post_ts
FROM bluesky
WHERE variant_get(data, 'kind', 'Utf8') = 'commit'
  AND variant_get(data, 'commit.operation', 'Utf8') = 'create'
  AND variant_get(data, 'commit.collection', 'Utf8') = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY first_post_ts ASC
LIMIT 3;

