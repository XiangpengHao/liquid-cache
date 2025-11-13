-- Q5 - top 3 users with longest activity
SELECT
    variant_get(data, 'did', 'string') AS user_id,
    CAST(EXTRACT(EPOCH FROM (MAX(TO_TIMESTAMP_MICROS(variant_get(data, 'time_us', 'bigint'))) - MIN(TO_TIMESTAMP_MICROS(variant_get(data, 'time_us', 'bigint'))))) * 1000 AS BIGINT) AS activity_span
FROM bluesky
WHERE variant_get(data, 'kind', 'string') = 'commit'
  AND variant_get(data, 'commit.operation', 'string') = 'create'
  AND variant_get(data, 'commit.collection', 'string') = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY activity_span DESC
LIMIT 3;
